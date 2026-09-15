"""Local training - the loop a single client runs on its own slice of data."""

import logging
import time
from typing import Optional

import torch
from transformers import get_linear_schedule_with_warmup

from .modeling import autocast_dtype

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

    # Mixed precision: fp32 master weights, fp16 compute. The scaler keeps small
    # gradients from underflowing fp16 on the way back.
    use_amp = device.type == 'cuda'
    amp_dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

    start = time.time()
    total_loss, num_batches, nonfinite = 0.0, 0, 0

    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)

            if not torch.isfinite(loss):
                # Silently training on NaN produces a model that predicts one
                # class and metrics that look merely bad rather than broken.
                nonfinite += 1
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

    if nonfinite:
        logger.error(f"{nonfinite} of {nonfinite + num_batches} batches produced a "
                     f"non-finite loss and were skipped - results are not trustworthy")

    return {
        'loss': total_loss / max(1, num_batches),
        'steps': num_batches,
        'nonfinite_batches': nonfinite,
        'seconds': time.time() - start,
    }
