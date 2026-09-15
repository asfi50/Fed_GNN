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
    """fp16/bf16 only pay off on CUDA; MPS is more reliable in fp32."""
    if device.type != 'cuda':
        return torch.float32

    if preferred == 'bfloat16':
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # fp32 would be the numerically safer fallback, but models that ship in
        # bf16 are too big to hold at 4 bytes per weight on a 16GB card:
        # gemma-4-E2B stores 5.1B weights, which is 20GB in fp32. fp16 is the
        # only format that fits. Loss is computed in fp32 (see client.py) to
        # contain the overflow risk that motivated bf16 in the first place.
        logger.warning("bfloat16 unsupported on this GPU; using fp16 with an fp32 loss")
        return torch.float16

    return torch.float16


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
    model.to(device)

    trainable, total = count_parameters(model)
    logger.info(f"Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.3f}%)")

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
