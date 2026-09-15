"""Local training - the loop a single client runs on its own slice of data."""

import logging
import time
from typing import Optional

import torch
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def train_local(
    model,
    loader,
    cfg,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    epochs: Optional[int] = None,
) -> dict:
    epochs = epochs if epochs is not None else cfg.federated.local_epochs
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.model.lr, weight_decay=cfg.training.weight_decay)

    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.training.warmup_ratio),
        num_training_steps=total_steps,
    )

    weights = class_weights.to(device) if class_weights is not None else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    start = time.time()
    total_loss, num_batches = 0.0, 0

    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')

            logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

    return {
        'loss': total_loss / max(1, num_batches),
        'steps': num_batches,
        'seconds': time.time() - start,
    }
