"""Server-side aggregation of client LoRA updates.

The subtlety this module exists for: a LoRA update is a product, dW = B @ A.
Averaging A and B separately is NOT the same as averaging the updates, because

    mean(B_i @ A_i) != mean(B_i) @ mean(A_i)

Every strategy below is a different answer to that problem, which is why they
are worth comparing rather than picking one up front.

    naive       average A and B independently (FedIT). Standard baseline, cheap,
                mathematically wrong but works well enough in practice.
    delta_svd   average the true dW = B @ A, then re-factorise back to rank r via
                SVD. The average of n rank-r updates has rank up to n*r, so this
                is the best rank-r approximation of the true average rather than
                the average itself - exact only when the average happens to fit
                in rank r. Costs one SVD per adapted layer per round.
    ffa         freeze A at its shared init and only train/average B. Since every
                client then holds the same A, mean(B_i @ A) = mean(B_i) @ A, so
                averaging really is exact here, and upload size halves. The cost
                is capacity: half the LoRA parameters no longer adapt.
    performance average weighted by each client's validation score rather than
                its sample count, mirroring the weighting used on the GNN side.
"""

import logging
import re
from typing import Dict, List, Sequence

import torch

logger = logging.getLogger(__name__)

STRATEGIES = ('naive', 'delta_svd', 'ffa', 'performance')

LORA_A = re.compile(r'\.lora_A\.')
LORA_B = re.compile(r'\.lora_B\.')


def aggregate(
    strategy: str,
    client_states: List[Dict[str, torch.Tensor]],
    sample_counts: Sequence[int],
    scores: Sequence[float] = None,
) -> Dict[str, torch.Tensor]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown aggregation strategy '{strategy}'. Options: {STRATEGIES}")

    if strategy == 'performance':
        if scores is None:
            raise ValueError("The 'performance' strategy needs per-client validation scores")
        weights = _normalise(scores)
    else:
        weights = _normalise(sample_counts)

    if strategy == 'delta_svd':
        return _aggregate_delta_svd(client_states, weights)
    if strategy == 'ffa':
        return _aggregate_ffa(client_states, weights)
    return _weighted_mean(client_states, weights)


def _normalise(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [float(v) / total for v in values]


def _weighted_mean(states: List[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Plain weighted average of every tensor, keys assumed identical across clients."""
    out = {}
    for key in states[0]:
        stacked = torch.stack([s[key].float() * w for s, w in zip(states, weights)])
        out[key] = stacked.sum(dim=0).to(states[0][key].dtype)
    return out


def _aggregate_ffa(states: List[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Average B (and the head) but keep A exactly as the first client has it.

    Clients are handed the same A every round and never update it, so all A
    tensors are already identical - averaging them would be a no-op that only
    risks drift from numerical noise.
    """
    out = _weighted_mean(states, weights)
    for key in states[0]:
        if LORA_A.search(key):
            out[key] = states[0][key].clone()
    return out


def _aggregate_delta_svd(states: List[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Average the real update B@A, then factorise back to rank r.

    The averaged delta can have rank up to n_clients * r, so truncating to r
    gives the closest rank-r update rather than an exact reconstruction. Keys
    that are not part of a LoRA pair (the classification head) fall through to a
    plain weighted average.
    """
    out = _weighted_mean(states, weights)

    for a_key in [k for k in states[0] if LORA_A.search(k)]:
        b_key = LORA_A.sub('.lora_B.', a_key)
        if b_key not in states[0]:
            continue

        rank = states[0][a_key].shape[0]
        delta = None
        for state, weight in zip(states, weights):
            update = state[b_key].float() @ state[a_key].float()
            delta = update * weight if delta is None else delta + update * weight

        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        if s.numel() < rank:
            # Truncating here would silently hand back factors of the wrong shape,
            # which only surfaces later as a corrupted load into the model.
            raise ValueError(
                f"LoRA rank {rank} exceeds the {tuple(delta.shape)} layer's maximum rank "
                f"{s.numel()} for '{a_key}'. Lower lora.r in the model config."
            )
        root_s = torch.sqrt(s[:rank])
        # Split the singular values evenly between the two factors so neither
        # side carries the whole magnitude
        out[b_key] = (u[:, :rank] * root_s).to(states[0][b_key].dtype)
        out[a_key] = (root_s.unsqueeze(1) * vh[:rank, :]).to(states[0][a_key].dtype)

    return out


def freeze_lora_a(model):
    """Used by the 'ffa' strategy: make the A matrices non-trainable."""
    frozen = 0
    for name, param in model.named_parameters():
        if LORA_A.search(name):
            param.requires_grad = False
            frozen += 1

    if frozen == 0:
        # Left unchecked this degrades FFA into plain FedAvg while still being
        # reported as FFA - a silently wrong result rather than a crash.
        raise RuntimeError(
            "FFA-LoRA froze no A matrices: no parameter name matched '.lora_A.'. "
            "The adapter layout is not what this strategy assumes."
        )
    logger.info(f"FFA-LoRA: froze {frozen} A matrices, only B is trained and communicated")


def upload_size_mb(state: Dict[str, torch.Tensor], strategy: str) -> float:
    """What a client actually sends. FFA skips A, so it uploads roughly half."""
    keys = state.keys()
    if strategy == 'ffa':
        keys = [k for k in keys if not LORA_A.search(k)]
    return sum(state[k].numel() * state[k].element_size() for k in keys) / (1024 ** 2)
