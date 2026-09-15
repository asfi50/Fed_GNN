"""Local training - the loop a single client runs on its own slice of data."""

import logging
import time
from typing import Optional

import torch
from transformers import get_linear_schedule_with_warmup

from .modeling import amp_context, amp_settings

logger = logging.getLogger(__name__)


def train_local(
    model,
    loader,
    cfg,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    epochs: Optional[int] = None,
    label: str = '',
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

    # Mixed precision: fp32 trainables, half-precision compute. The scaler keeps
    # small gradients from underflowing fp16 on the way back; with a bf16 or fp32
    # base there is nothing to scale.
    amp_on, amp_dtype = amp_settings(model, device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_on and amp_dtype == torch.float16)

    start = time.time()
    total_loss, num_batches, nonfinite = 0.0, 0, 0
    # Report a handful of times per client so a long round is not silent, without
    # flooding the log the way a per-batch bar would once piped to a notebook.
    report_every = max(1, total_steps // 4)

    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')

            with amp_context(model, device):
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

            if label and num_batches % report_every == 0:
                pct = 100 * num_batches / total_steps
                logger.info(f"      {label} {pct:3.0f}%  ({num_batches}/{total_steps} steps)  "
                            f"loss={total_loss / num_batches:.4f}")

    if nonfinite:
        logger.error(f"{nonfinite} of {nonfinite + num_batches} batches produced a "
                     f"non-finite loss and were skipped - results are not trustworthy")

    return {
        'loss': total_loss / max(1, num_batches),
        'steps': num_batches,
        'nonfinite_batches': nonfinite,
        'seconds': time.time() - start,
    }
