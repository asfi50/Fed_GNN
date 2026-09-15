"""Evaluation metrics, matched to the ones the GNN experiments report."""

import logging
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .modeling import autocast_dtype

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, loader, device, label_names: List[str], max_batches: int = None) -> Dict:
    """max_batches caps the evaluation - used for the cheap per-client scoring that
    performance-weighted aggregation needs every round."""
    model.eval()
    all_preds, all_labels = [], []

    use_amp = device.type == 'cuda'
    amp_dtype = autocast_dtype(device)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        labels = batch.pop('labels')
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(**batch).logits
        all_preds.append(logits.float().argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    class_ids = list(range(len(label_names)))

    _, _, per_class_f1, support = precision_recall_fscore_support(
        labels, preds, labels=class_ids, zero_division=0
    )

    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
        'macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(labels, preds, average='weighted', zero_division=0)),
        'per_class_f1': {name: float(f) for name, f in zip(label_names, per_class_f1)},
        'support': {name: int(s) for name, s in zip(label_names, support)},
        'confusion_matrix': confusion_matrix(labels, preds, labels=class_ids).tolist(),
        'num_samples': int(len(labels)),
    }


def log_metrics(metrics: Dict, prefix: str = ''):
    logger.info(
        f"{prefix}acc={metrics['accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )


def log_per_class(metrics: Dict):
    logger.info("Per-class F1:")
    for name, score in sorted(metrics['per_class_f1'].items(), key=lambda kv: -kv[1]):
        logger.info(f"  {name:<12} f1={score:.4f}  (n={metrics['support'][name]})")
