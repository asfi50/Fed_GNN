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


def resolve_device(requested: str = 'auto') -> torch.device:
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_dtype(device: torch.device, preferred: str = None) -> torch.dtype:
    """Precision of the FROZEN base weights.

    Half precision is free here: LoRA never updates the base, so it cannot
    accumulate rounding error, and halving it leaves room for activations. What
    must not be half precision is anything being optimised - AdamW's
    second-moment estimate underflows fp16 and the weights walk into NaN - so
    build_model upcasts every trainable tensor to fp32 afterwards.
    """
    if device.type != 'cuda':
        return torch.float32
    if preferred == 'bfloat16' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Compute precision inside autocast. bf16 where available, else fp16."""
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def amp_context(device: torch.device):
    """Every forward pass must run inside this.

    The base is half precision while the adapters and head are fp32, so an
    unwrapped call hits 'mat1 and mat2 must have the same dtype'. Autocast is
    what reconciles them, which is why this lives in one place rather than being
    repeated at each call site.
    """
    return torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype(device),
        enabled=device.type == 'cuda',
    )


def build_model(cfg, num_labels: int, device: torch.device) -> Tuple[torch.nn.Module, object]:
    model_cfg = cfg.model
    logger.info(f"Loading {model_cfg.hf_id} ({num_labels} labels)...")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)
    dtype = resolve_dtype(device, model_cfg.get('preferred_dtype'))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.hf_id,
        num_labels=num_labels,
        torch_dtype=dtype,
    )

    if model_cfg.get('needs_pad_token'):
        # Decoder models ship without a pad token, but the classification head
        # needs one to find the last real token of each sequence.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

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

    trainable, total = count_parameters(model)
    logger.info(f"Base dtype {dtype}, trainable: {trainable:,} / {total:,} "
                f"params ({100 * trainable / total:.3f}%)")

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
