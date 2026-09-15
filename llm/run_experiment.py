"""Run one federated experiment: one model, one split, one aggregation strategy.

Usage (from the repo root):
    python llm/run_experiment.py --model bert --split iid
    python llm/run_experiment.py --model qwen --split non_iid --agg delta_svd

On Kaggle, run the two splits in parallel across the two T4s:
    CUDA_VISIBLE_DEVICES=0 python llm/run_experiment.py --model qwen --split iid &
    CUDA_VISIBLE_DEVICES=1 python llm/run_experiment.py --model qwen --split non_iid &
    wait
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fedllm.aggregation import STRATEGIES  # noqa: E402
from fedllm.config import load_config  # noqa: E402
from fedllm.data import (  # noqa: E402
    compute_class_weights,
    frame_to_dataset,
    load_client_shards,
    load_eval_frame,
    load_label_mapper,
    make_loader,
)
from fedllm.evaluate import evaluate, log_metrics, log_per_class  # noqa: E402
from fedllm.modeling import build_model, count_parameters, resolve_device  # noqa: E402
from fedllm.server import FederatedServer  # noqa: E402
from fedllm.tracking import collect_parameters, init_tracker, load_split_parameters  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Federated LoRA fine-tuning')
    parser.add_argument('--model', type=str, required=True, help='Config name: bert, t5, llama, qwen, gemma')
    parser.add_argument('--split', type=str, default='iid', choices=['iid', 'non_iid'])
    parser.add_argument('--agg', type=str, default=None, choices=list(STRATEGIES))
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--rows_per_round', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', action='store_true', help='Continue from the last saved round')
    parser.add_argument('--no_push', action='store_true', help='Skip uploading the adapter to the Hub')
    parser.add_argument('--no_comet', action='store_true', help='Skip Comet ML tracking')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.model, overrides={
        'data.data_dir': args.data_dir,
        'output.output_dir': args.output_dir,
        'federated.aggregation': args.agg,
        'federated.num_rounds': args.rounds,
        'federated.rows_per_client_per_round': args.rows_per_round,
        'model.batch_size': args.batch_size,
        'output.push_to_hub': False if args.no_push else None,
    })

    torch.manual_seed(cfg.seed)
    device = resolve_device(args.device)
    run_name = f"{cfg.model.name}_{args.split}_{cfg.federated.aggregation}"
    out_dir = os.path.join(cfg.output.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Run: {run_name}  |  device={device}  |  {cfg.model.hf_id}")

    tracker = init_tracker(
        run_name,
        tags=[cfg.model.name, args.split, cfg.federated.aggregation, 'federated'],
        disabled=args.no_comet,
    )

    label_mapper = load_label_mapper(cfg.data.data_dir)
    label_names = [name for name, _ in sorted(label_mapper.items(), key=lambda kv: kv[1])]

    model, tokenizer = build_model(cfg, len(label_names), device)
    trainable, total = count_parameters(model)

    shards = load_client_shards(cfg, args.split, label_mapper)

    params = collect_parameters(cfg, args, extra={
        'run_mode': 'federated',
        'device': str(device),
        'trainable_params': trainable,
        'total_params': total,
        'trainable_pct': round(100 * trainable / total, 4),
        'client_shard_sizes': ','.join(str(len(s)) for s in shards),
    })
    params.update(load_split_parameters(cfg.data.data_dir))
    tracker.log_parameters(params)
    val_df = load_eval_frame(cfg, 'val', cfg.evaluation.val_subset, label_mapper)
    val_loader = make_loader(frame_to_dataset(val_df, cfg, label_mapper), tokenizer, cfg, shuffle=False)

    class_weights = (
        compute_class_weights(cfg.data.data_dir, label_mapper, cfg.training.max_class_weight)
        if cfg.training.use_class_weights else None
    )

    server = FederatedServer(model, tokenizer, cfg, device, label_names, class_weights, tracker=tracker)
    if args.resume:
        server.load_checkpoint(out_dir)

    start = time.time()
    history = server.run(shards, val_loader, checkpoint_dir=out_dir)

    logger.info("Final evaluation on the held-out test set...")
    test_df = load_eval_frame(cfg, 'test', cfg.evaluation.test_subset, label_mapper)
    test_loader = make_loader(frame_to_dataset(test_df, cfg, label_mapper), tokenizer, cfg, shuffle=False)
    metrics = evaluate(model, test_loader, device, label_names)
    log_metrics(metrics, prefix='TEST  ')
    log_per_class(metrics)

    results = {
        'run': run_name,
        'model': cfg.model.name,
        'hf_id': cfg.model.hf_id,
        'split': args.split,
        'aggregation': cfg.federated.aggregation,
        'num_clients': cfg.federated.num_clients,
        'num_rounds': cfg.federated.num_rounds,
        'rows_per_client_per_round': cfg.federated.rows_per_client_per_round,
        'trainable_params': trainable,
        'total_params': total,
        'client_shard_sizes': [len(s) for s in shards],
        'total_seconds': time.time() - start,
        'communication_mb': sum(r['upload_mb_per_client'] * cfg.federated.num_clients for r in history),
        'history': history,
        'test_metrics': metrics,
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {out_dir}/results.json")

    tracker.log_metrics(
        {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, prefix='test_'
    )
    tracker.log_metrics({f'test_f1_{name}': f for name, f in metrics['per_class_f1'].items()})
    tracker.log_metrics({
        'total_seconds': results['total_seconds'],
        'communication_mb': results['communication_mb'],
    })
    tracker.log_confusion_matrix(metrics['confusion_matrix'], label_names)
    tracker.log_asset(os.path.join(out_dir, 'results.json'))

    if cfg.output.save_adapters:
        model.save_pretrained(os.path.join(out_dir, 'adapter'))
        logger.info(f"Saved adapter to {out_dir}/adapter")

    if cfg.output.push_to_hub:
        push_adapter(model, cfg, tracker)

    tracker.end()


def push_adapter(model, cfg, tracker):
    token = os.environ.get('HF_TOKEN')
    if not token:
        logger.warning("push_to_hub is on but HF_TOKEN is not set; skipping upload")
        return

    repo_id = f"{cfg.output.hub_prefix}-{cfg.model.hf_id.split('/')[-1]}"
    try:
        model.push_to_hub(repo_id, token=token)
        logger.info(f"Pushed adapter to the Hub as {repo_id}")
        tracker.log_parameters({'hub_repo': repo_id})
    except Exception as e:
        # By this point training is done and the results are on disk. A bad token
        # or a network blip must not discard hours of GPU time at the finish line.
        logger.error(f"Hub upload to '{repo_id}' failed: {e}")
        logger.error("Results and the adapter are saved locally; upload them later "
                     "from the run's adapter/ directory.")


if __name__ == '__main__':
    main()
