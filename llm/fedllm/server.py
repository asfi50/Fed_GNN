"""The federated round loop: broadcast, local training, aggregate, evaluate.

Clients are simulated sequentially on one GPU. Only one copy of the base model
ever exists - clients differ only by which LoRA state is loaded into it, which
is what makes five 2B-parameter clients fit in 16GB.
"""

import json
import logging
import os
import time
from typing import Dict, List

import torch

from .aggregation import aggregate, freeze_lora_a, upload_size_mb
from .client import train_local
from .data import make_loader
from .evaluate import evaluate, log_metrics
from .modeling import get_trainable_state, set_trainable_state

logger = logging.getLogger(__name__)


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    return f'{seconds // 3600}h{(seconds % 3600) // 60:02d}m'


class FederatedServer:
    def __init__(self, model, tokenizer, cfg, device, label_names, class_weights=None, tracker=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.label_names = label_names
        self.class_weights = class_weights
        self.tracker = tracker
        self.strategy = cfg.federated.aggregation

        if self.strategy == 'ffa':
            freeze_lora_a(model)

        self.global_state = get_trainable_state(model)
        self.history: List[Dict] = []
        self.start_round = 0

    def run(self, shards, val_loader, checkpoint_dir: str) -> List[Dict]:
        os.makedirs(checkpoint_dir, exist_ok=True)
        num_rounds = self.cfg.federated.num_rounds
        num_clients = len(shards)
        run_start = time.time()

        for round_idx in range(self.start_round, num_rounds):
            round_start = time.time()
            pct = 100 * round_idx / num_rounds
            logger.info(f"{'=' * 64}")
            logger.info(f"Round {round_idx + 1}/{num_rounds}  [{pct:3.0f}% complete]  "
                        f"aggregation={self.strategy}")

            client_states, sample_counts, client_losses = [], [], []
            client_scores = [] if self.strategy == 'performance' else None

            for position, shard in enumerate(shards, start=1):
                # Broadcast: every client starts the round from the same global state
                set_trainable_state(self.model, self.global_state)

                dataset = shard.round_dataset(round_idx)
                loader = make_loader(dataset, self.tokenizer, self.cfg, shuffle=True)
                stats = train_local(
                    self.model, loader, self.cfg, self.device, self.class_weights,
                    label=f'client {position}/{num_clients}',
                )

                client_states.append(get_trainable_state(self.model))
                sample_counts.append(len(dataset))
                client_losses.append(stats['loss'])

                score_msg = ''
                if client_scores is not None:
                    # Performance weighting needs to know how good each client's
                    # update actually is, scored on a slice of the shared val set
                    score = evaluate(
                        self.model, val_loader, self.device, self.label_names,
                        max_batches=self.cfg.evaluation.client_score_batches,
                    )['macro_f1']
                    client_scores.append(score)
                    score_msg = f", val_macro_f1={score:.4f}"

                logger.info(f"  client {shard.client_id}: {len(dataset)} rows, "
                            f"loss={stats['loss']:.4f}, {stats['seconds']:.0f}s{score_msg}")

            self.global_state = aggregate(self.strategy, client_states, sample_counts, scores=client_scores)
            set_trainable_state(self.model, self.global_state)

            record = {
                'round': round_idx + 1,
                'client_losses': client_losses,
                'mean_client_loss': sum(client_losses) / len(client_losses),
                'client_scores': client_scores,
                'sample_counts': sample_counts,
                'upload_mb_per_client': upload_size_mb(client_states[0], self.strategy),
                'seconds': time.time() - round_start,
            }

            if (round_idx + 1) % self.cfg.evaluation.eval_every == 0 or round_idx == num_rounds - 1:
                metrics = evaluate(self.model, val_loader, self.device, self.label_names)
                record['val'] = metrics
                log_metrics(metrics, prefix=f"  round {round_idx + 1} VAL   ")

            self._track_round(record, round_idx + 1)
            self.history.append(record)
            self.save_checkpoint(checkpoint_dir, round_idx + 1)

            done = round_idx + 1 - self.start_round
            todo = num_rounds - self.start_round
            elapsed = time.time() - run_start
            eta = elapsed / done * (todo - done)
            logger.info(f"  round done in {record['seconds']:.0f}s  |  "
                        f"{100 * (round_idx + 1) / num_rounds:3.0f}% complete  |  "
                        f"elapsed {_hms(elapsed)}  |  eta {_hms(eta)}  |  "
                        f"upload {record['upload_mb_per_client']:.1f} MB/client")

        return self.history

    def _track_round(self, record: Dict, step: int):
        if self.tracker is None:
            return

        self.tracker.log_metrics({
            'mean_client_loss': record['mean_client_loss'],
            'round_seconds': record['seconds'],
            'upload_mb_per_client': record['upload_mb_per_client'],
        }, step=step)

        self.tracker.log_metrics(
            {f'client_{i + 1}_loss': loss for i, loss in enumerate(record['client_losses'])},
            step=step,
        )

        if record.get('client_scores'):
            self.tracker.log_metrics(
                {f'client_{i + 1}_score': s for i, s in enumerate(record['client_scores'])},
                step=step,
            )

        if 'val' in record:
            self.tracker.log_metrics(
                {k: v for k, v in record['val'].items() if isinstance(v, (int, float))},
                step=step,
                prefix='val_',
            )

    def save_checkpoint(self, checkpoint_dir: str, completed_rounds: int):
        """Kaggle sessions cap out at 9 hours, so every round is resumable."""
        torch.save(
            {'global_state': self.global_state, 'completed_rounds': completed_rounds, 'history': self.history},
            os.path.join(checkpoint_dir, 'checkpoint.pt'),
        )
        with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, checkpoint_dir: str) -> bool:
        path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.exists(path):
            return False

        ckpt = torch.load(path, map_location='cpu')
        self.global_state = ckpt['global_state']
        self.history = ckpt['history']
        self.start_round = ckpt['completed_rounds']
        set_trainable_state(self.model, self.global_state)
        logger.info(f"Resumed from checkpoint at round {self.start_round}")
        return True
