"""Comet ML tracking for the fed-llm-ids project.

Every run logs the full parameter set - model, LoRA config, dataset provenance,
split parameters, federated schedule - so a result in the Comet UI can always be
traced back to the exact configuration that produced it.

Logging failures never abort a run: losing a metric is not worth losing five
hours of GPU time.
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

COMET_API_KEY = "emuhTVn5AAwEm9ALtwsL4SkUo"
PROJECT_NAME = "fed-llm-ids"


class Tracker:
    """Thin wrapper over a Comet experiment. Becomes a no-op if Comet is unavailable."""

    def __init__(self, experiment=None):
        self.exp = experiment

    @property
    def enabled(self) -> bool:
        return self.exp is not None

    def log_parameters(self, params: Dict):
        if not self.enabled:
            return
        try:
            self.exp.log_parameters(params)
        except Exception as e:
            logger.warning(f"Comet log_parameters failed: {e}")

    def log_metrics(self, metrics: Dict, step: Optional[int] = None, prefix: str = ''):
        if not self.enabled:
            return
        try:
            flat = {f"{prefix}{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.exp.log_metrics(flat, step=step)
        except Exception as e:
            logger.warning(f"Comet log_metrics failed: {e}")

    def log_confusion_matrix(self, matrix: List[List[int]], labels: List[str]):
        if not self.enabled:
            return
        try:
            self.exp.log_confusion_matrix(matrix=matrix, labels=labels)
        except Exception as e:
            logger.warning(f"Comet log_confusion_matrix failed: {e}")

    def log_asset(self, path: str):
        if not self.enabled or not os.path.exists(path):
            return
        try:
            self.exp.log_asset(path)
        except Exception as e:
            logger.warning(f"Comet log_asset failed: {e}")

    def add_tags(self, tags: List[str]):
        if not self.enabled:
            return
        try:
            self.exp.add_tags(tags)
        except Exception as e:
            logger.warning(f"Comet add_tags failed: {e}")

    def end(self):
        if not self.enabled:
            return
        try:
            self.exp.end()
        except Exception as e:
            logger.warning(f"Comet end failed: {e}")


def init_tracker(run_name: str, tags: List[str], disabled: bool = False) -> Tracker:
    if disabled:
        logger.info("Comet tracking disabled")
        return Tracker(None)

    try:
        import comet_ml

        comet_ml.login(api_key=COMET_API_KEY)
        exp = comet_ml.start(project_name=PROJECT_NAME)
        exp.set_name(run_name)
        exp.add_tags(tags)
        logger.info(f"Comet experiment '{run_name}' started in project '{PROJECT_NAME}'")
        return Tracker(exp)
    except Exception as e:
        logger.warning(f"Could not start Comet tracking ({e}); continuing without it")
        return Tracker(None)


def collect_parameters(cfg, args, extra: Dict = None) -> Dict:
    """Everything needed to reproduce a run, flattened for the Comet params table."""
    model = cfg.model
    params = {
        'run_mode': extra.get('run_mode', 'federated') if extra else 'federated',
        'split': args.split,
        'seed': cfg.seed,

        'model_name': model.name,
        'model_hf_id': model.hf_id,
        'model_family': model.get('family'),
        'max_length': model.max_length,
        'batch_size': model.batch_size,
        'learning_rate': model.lr,

        'lora_r': model.lora.r,
        'lora_alpha': model.lora.alpha,
        'lora_dropout': model.lora.dropout,
        'lora_target_modules': ','.join(model.lora.target_modules),

        'num_clients': cfg.federated.num_clients,
        'num_rounds': cfg.federated.num_rounds,
        'local_epochs': cfg.federated.local_epochs,
        'rows_per_client_per_round': cfg.federated.rows_per_client_per_round,
        'aggregation': cfg.federated.aggregation,

        'weight_decay': cfg.training.weight_decay,
        'grad_clip': cfg.training.grad_clip,
        'warmup_ratio': cfg.training.warmup_ratio,
        'use_class_weights': cfg.training.use_class_weights,
        'max_class_weight': cfg.training.max_class_weight,

        'val_subset': cfg.evaluation.val_subset,
        'test_subset': cfg.evaluation.test_subset,

        'drop_columns': ','.join(cfg.data.drop_columns),
        'include_ips': cfg.data.include_ips,
        'data_dir': cfg.data.data_dir,
    }
    params.update(extra or {})
    return params


def load_split_parameters(data_dir: str) -> Dict:
    """Pull the dataset/split provenance written by split_dataset.py."""
    path = os.path.join(data_dir, 'split_config.json')
    if not os.path.exists(path):
        logger.warning(f"{path} not found; split provenance will not be logged")
        return {}

    with open(path) as f:
        split_cfg = json.load(f)
    return {f'split_{k}': v for k, v in split_cfg.items() if not isinstance(v, (dict, list))}
