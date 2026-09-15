"""Centralized baseline - the upper bound the federated runs are measured against.

Trains one model on the pooled data from all clients, with exactly the same data
budget a federated run gets (num_clients x rows_per_round x num_rounds rows, one
pass). Any gap between this and the federated result is the cost of federation.

Usage:
    python llm/run_centralized.py --model bert
    python llm/run_centralized.py --model bert --rows_per_round 500 --rounds 2
"""

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fedllm.client import train_local  # noqa: E402
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
from fedllm.tracking import collect_parameters, init_tracker, load_split_parameters  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Centralized (non-federated) baseline')
    parser.add_argument('--model', type=str, required=True, help='Config name: bert, t5, llama, qwen, gemma')
    parser.add_argument('--split', type=str, default='iid', choices=['iid', 'non_iid'],
                        help='Which partition to pool from (the union is the same either way)')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--rounds', type=int, default=None,
                        help='Rounds worth of data budget to match (default: config)')
    parser.add_argument('--rows_per_round', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--no_comet', action='store_true', help='Skip Comet ML tracking')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.model, overrides={
        'data.data_dir': args.data_dir,
        'output.output_dir': args.output_dir,
        'federated.num_rounds': args.rounds,
        'federated.rows_per_client_per_round': args.rows_per_round,
        'model.batch_size': args.batch_size,
    })

    torch.manual_seed(cfg.seed)
    device = resolve_device(args.device)
    logger.info(f"Device: {device}")

    label_mapper = load_label_mapper(cfg.data.data_dir)
    label_names = [name for name, _ in sorted(label_mapper.items(), key=lambda kv: kv[1])]
    logger.info(f"{len(label_names)} classes: {label_names}")

    # Pool exactly the rows a federated run would consume, so the two are comparable
    shards = load_client_shards(cfg, args.split, label_mapper)
    frames = [shard.round_frame(r) for shard in shards for r in range(cfg.federated.num_rounds)]
    pooled = pd.concat(frames).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    logger.info(f"Pooled training set: {len(pooled)} rows "
                f"({cfg.federated.num_clients} clients x {cfg.federated.rows_per_client_per_round} "
                f"rows x {cfg.federated.num_rounds} rounds)")

    model, tokenizer = build_model(cfg, len(label_names), device)
    trainable, total = count_parameters(model)

    tracker = init_tracker(
        f'centralized_{cfg.model.name}_{args.split}',
        tags=[cfg.model.name, args.split, 'centralized', 'baseline'],
        disabled=args.no_comet,
    )
    params = collect_parameters(cfg, args, extra={
        'run_mode': 'centralized',
        'device': str(device),
        'trainable_params': trainable,
        'total_params': total,
        'pooled_train_rows': len(pooled),
    })
    params.update(load_split_parameters(cfg.data.data_dir))
    tracker.log_parameters(params)

    train_loader = make_loader(frame_to_dataset(pooled, cfg, label_mapper), tokenizer, cfg, shuffle=True)
    test_df = load_eval_frame(cfg, 'test', cfg.evaluation.test_subset, label_mapper)
    test_loader = make_loader(frame_to_dataset(test_df, cfg, label_mapper), tokenizer, cfg, shuffle=False)

    class_weights = (
        compute_class_weights(cfg.data.data_dir, label_mapper, cfg.training.max_class_weight)
        if cfg.training.use_class_weights else None
    )

    logger.info("Training...")
    start = time.time()
    stats = train_local(model, train_loader, cfg, device, class_weights, epochs=1)
    logger.info(f"Done in {stats['seconds']:.0f}s over {stats['steps']} steps, loss={stats['loss']:.4f}")

    logger.info("Evaluating on the held-out test set...")
    metrics = evaluate(model, test_loader, device, label_names)
    log_metrics(metrics, prefix='TEST  ')
    log_per_class(metrics)

    out_dir = os.path.join(cfg.output.output_dir, f'centralized_{cfg.model.name}_{args.split}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({
            'mode': 'centralized',
            'model': cfg.model.name,
            'hf_id': cfg.model.hf_id,
            'split': args.split,
            'train_rows': len(pooled),
            'train_seconds': stats['seconds'],
            'train_loss': stats['loss'],
            'total_seconds': time.time() - start,
            'metrics': metrics,
        }, f, indent=2)
    logger.info(f"Saved results to {out_dir}/results.json")

    tracker.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, prefix='test_')
    tracker.log_metrics({f'test_f1_{name}': f for name, f in metrics['per_class_f1'].items()})
    tracker.log_metrics({'train_loss': stats['loss'], 'train_seconds': stats['seconds']})
    tracker.log_confusion_matrix(metrics['confusion_matrix'], label_names)
    tracker.log_asset(os.path.join(out_dir, 'results.json'))
    tracker.end()


if __name__ == '__main__':
    main()
