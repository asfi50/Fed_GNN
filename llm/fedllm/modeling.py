"""One model-building path for every architecture in the study.

BERT, T5, Llama, Qwen and Gemma all expose an AutoModelForSequenceClassification
variant, so all five run through an identical loss, metric and LoRA setup. That
is what keeps the cross-model comparison honest - no per-family decoding rules,
no output parsing, no invalid predictions.
"""

import logging
from typing import Dict, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def _transformers_version() -> str:
    import transformers
    return transformers.__version__


def resolve_device(requested: str = 'auto') -> torch.device:
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_dtype(device: torch.device, model_cfg) -> torch.dtype:
    """Precision of the FROZEN base weights.

    Half precision is normally free: LoRA never updates the base, so it cannot
    accumulate rounding error, and halving it leaves room for activations. What
    must not be half precision is anything being optimised - AdamW's
    second-moment estimate underflows fp16 and the weights walk into NaN - so
    build_model upcasts every trainable tensor to fp32 afterwards.

    Models setting `precision: fp32` opt out entirely. T5 has to: it was trained
    in bf16 and fills its attention mask with finfo(fp32).min, which overflows
    fp16 outright. Small models pay nothing for fp32 anyway.
    """
    if device.type != 'cuda' or model_cfg.get('precision') == 'fp32':
        return torch.float32
    if model_cfg.get('preferred_dtype') == 'bfloat16' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def amp_settings(model, device: torch.device):
    """(enabled, dtype) for autocast, derived from how the base was loaded.

    Mixed precision is on exactly when the base is half precision, which is also
    exactly when it is needed: a half base and an fp32 head can only meet inside
    autocast. An fp32 base needs nothing and gets nothing.
    """
    dtype = getattr(model, 'fedllm_base_dtype', torch.float32)
    enabled = device.type == 'cuda' and dtype in (torch.float16, torch.bfloat16)
    return enabled, dtype


def amp_context(model, device: torch.device):
    """Every forward pass must run inside this - see amp_settings."""
    enabled, dtype = amp_settings(model, device)
    return torch.autocast(
        device_type=device.type,
        dtype=dtype if enabled else torch.float16,
        enabled=enabled,
    )


def ensure_pad_token(model, tokenizer):
    """Give the model a pad token id it can actually find.

    A classification head locates each sequence's last real token by counting
    padding, so without a pad id it refuses any batch larger than one. Decoder
    checkpoints often ship without one, and composite configs (Qwen3.5 and other
    multimodal models) read it from a nested text config rather than the top
    level - so it is written to every config that exposes the field, then read
    back to confirm it stuck.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError(
            f"{tokenizer.__class__.__name__} has neither a pad token nor an eos token to "
            "fall back on; this model needs an explicit pad token before it can be batched."
        )

    candidates = [model.config, getattr(model.config, 'text_config', None)]
    get_text_config = getattr(model.config, 'get_text_config', None)
    if callable(get_text_config):
        candidates.append(get_text_config())

    seen = set()
    for config in candidates:
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            config.pad_token_id = pad_id

    if getattr(model.config, 'pad_token_id', None) is None:
        raise RuntimeError("pad_token_id did not take on this model's config")
    logger.info(f"pad_token_id={pad_id} ({tokenizer.pad_token!r}), set on {len(seen)} config(s)")
    return pad_id


def build_model(cfg, num_labels: int, device: torch.device) -> Tuple[torch.nn.Module, object]:
    model_cfg = cfg.model
    logger.info(f"Loading {model_cfg.hf_id} ({num_labels} labels)...")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)
    dtype = resolve_dtype(device, model_cfg)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.hf_id,
            num_labels=num_labels,
            torch_dtype=dtype,
        )
    except ValueError as e:
        if 'Unrecognized configuration class' not in str(e):
            raise
        # transformers answers this with a hundred config names and no advice.
        # Every architecture here goes through the one AutoModel class, so a
        # model without that head cannot join the study as-is.
        raise RuntimeError(
            f"{model_cfg.hf_id} has no sequence-classification head in transformers "
            f"{_transformers_version()}. This pipeline routes every architecture through "
            f"AutoModelForSequenceClassification, so either pick a checkpoint from the same "
            f"family that does support it (newer multimodal releases often do not, while the "
            f"previous generation does), or give this model its own pooling head."
        ) from e

    ensure_pad_token(model, tokenizer)

    lora = model_cfg.lora
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias='none',
    )
    model = get_peft_model(model, peft_config)

    # Whatever the base is stored as, anything being optimised stays fp32.
    # Optimiser state in fp16 underflows and the model never learns; autocast
    # handles the mixed dtypes during the forward pass.
    upcast = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
    if upcast:
        logger.info(f"Upcast {len(upcast)} trainable tensors to fp32 (base stays {dtype})")

    model.to(device)
    # Read back by amp_settings to decide whether autocast is needed
    model.fedllm_base_dtype = dtype

    trainable, total = count_parameters(model)
    amp_on, _ = amp_settings(model, device)
    logger.info(f"Base dtype {dtype} (autocast {'on' if amp_on else 'off'}), trainable: "
                f"{trainable:,} / {total:,} params ({100 * trainable / total:.3f}%)")

    return model, tokenizer


def count_parameters(model) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_trainable_state(model) -> Dict[str, torch.Tensor]:
    """The payload a client sends to the server.

    Includes both the LoRA matrices and the classification head: the head starts
    from a random init, so averaging LoRA while leaving five different heads in
    place would make the aggregated model meaningless.
    """
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def set_trainable_state(model, state: Dict[str, torch.Tensor]):
    # peft rewrites the keys of the dict it is handed, in place. The server
    # broadcasts one global state to every client in turn, so without this copy
    # the second client would be handed a dict whose keys no longer resolve.
    set_peft_model_state_dict(model, dict(state))


def state_size_mb(state: Dict[str, torch.Tensor]) -> float:
    return sum(t.numel() * t.element_size() for t in state.values()) / (1024 ** 2)
