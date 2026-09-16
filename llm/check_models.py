"""Pre-flight check: prove the pipeline works for every model before spending GPU quota.

Run this first on Kaggle. A 2B model run costs ~5 hours, so it is worth two
minutes to confirm that each model loads, accepts the pipeline's batches, and
survives a full train/aggregate cycle.

Modes, each a superset of the previous:
    (default)   config and model id resolve on the Hub - no weights downloaded
    --load      weights download and the model builds
    --forward   the real path: LoRA attach, tokenised batch, forward, backward,
                LoRA state round-trip, and aggregation across two mock clients

Usage:
    python llm/check_models.py
    python llm/check_models.py --forward
    python llm/check_models.py --forward --models qwen gemma
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fedllm.config import CONFIG_DIR, load_config  # noqa: E402

NUM_LABELS = 10
SAMPLE_ROWS = [
    "sport=37446 dport=80 proto=6 l7=7.0 inb=112 outb=60 inp=2 outp=1 flags=18 dur=16",
    "sport=52690 dport=80 proto=6 l7=7.0 inb=734 outb=736 inp=5 outp=5 flags=27 dur=347",
    "sport=40134 dport=53 proto=17 l7=5.0 inb=67 outb=67 inp=1 outp=1 flags=0 dur=2",
    "sport=1234 dport=443 proto=6 l7=7.0 inb=99 outb=120 inp=3 outp=2 flags=22 dur=88",
]


def check_config_only(cfg):
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(cfg.model.hf_id)
    return f"OK  ({hf_cfg.model_type})"


def check_load(cfg):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(cfg.model.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.hf_id, num_labels=NUM_LABELS, torch_dtype=torch.float32
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    del model
    return f"OK  ({params:.0f}M params)"


def check_forward(cfg):
    """Exercise every step a real round performs, on four rows."""
    import torch

    from fedllm.aggregation import aggregate
    from fedllm.data import FlowDataset, make_collate_fn
    from fedllm.modeling import (
        amp_context, build_model, get_trainable_state, set_trainable_state, state_size_mb,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = build_model(cfg, NUM_LABELS, device)

    # Same collate path the training loop uses
    dataset = FlowDataset(SAMPLE_ROWS, [0, 2, 4, 1])
    collate = make_collate_fn(tokenizer, cfg.model.max_length)
    batch = collate([dataset[i] for i in range(len(dataset))])
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop('labels')

    # Same wrapper the training loop uses - the fp16 base and fp32 head only
    # meet correctly inside autocast
    with amp_context(model, device):
        logits = model(**batch).logits
    if logits.shape != (len(SAMPLE_ROWS), NUM_LABELS):
        raise RuntimeError(f"expected logits {(len(SAMPLE_ROWS), NUM_LABELS)}, got {tuple(logits.shape)}")

    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss.backward()

    if not torch.isfinite(loss):
        raise RuntimeError(f"loss is {loss.item()} on the very first batch")

    grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads:
        raise RuntimeError("backward produced no gradients on any trainable parameter")

    # LoRA state must survive a round-trip, since that is how clients are swapped
    state = get_trainable_state(model)
    set_trainable_state(model, state)

    # And the server must be able to aggregate it at this model's real dimensions
    other = {k: v.clone() for k, v in state.items()}
    for strategy in ('naive', 'delta_svd', 'ffa'):
        merged = aggregate(strategy, [state, other], [100, 200])
        set_trainable_state(model, merged)

    size = state_size_mb(state)
    del model
    return f"OK  (loss={loss.item():.3f}, {len(grads)} grad tensors, upload={size:.1f} MB)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help='Download weights and build each model')
    parser.add_argument('--forward', action='store_true',
                        help='Full pipeline check: LoRA, batch, forward, backward, aggregation')
    parser.add_argument('--models', nargs='*', default=None, help='Subset to check (default: all)')
    parser.add_argument('--verbose', action='store_true', help='Print full tracebacks on failure')
    args = parser.parse_args()

    names = args.models or sorted(
        f[:-5] for f in os.listdir(os.path.join(CONFIG_DIR, 'models')) if f.endswith('.yaml')
    )

    if args.forward:
        check, mode = check_forward, 'full pipeline'
    elif args.load:
        check, mode = check_load, 'model build'
    else:
        check, mode = check_config_only, 'config only'

    print(f"Checking {len(names)} models ({mode})\n")
    print(f"{'config':<8} {'hf_id':<30} status")
    print('-' * 78)

    failures = []
    for name in names:
        cfg = load_config(name)
        try:
            status = check(cfg)
        except Exception as e:
            status = f"FAIL  {type(e).__name__}: {str(e)[:95]}"
            failures.append(name)
            if args.verbose:
                traceback.print_exc()
        print(f"{name:<8} {cfg.model.hf_id:<30} {status}")

    if failures:
        print(f"\nFailed: {', '.join(failures)}  (rerun with --verbose for tracebacks)")
        print("A dead model id means the Hub moved on - check for the current release and "
              "update llm/configs/models/. A forward-pass failure usually means that "
              "architecture needs its own input handling rather than the shared path.")
        sys.exit(1)

    print(f"\nAll {len(names)} models passed.")


if __name__ == '__main__':
    main()
