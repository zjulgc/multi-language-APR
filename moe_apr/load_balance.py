"""Load-balance auxiliary loss for MoE-LoRA routing experts.

Two standard formulations exist:

1. "Switch Transformer" load-balance loss (Fedus et al. 2021):
       L_aux = num_experts * Σ_i (f_i * P_i)
   where:
     - f_i: fraction of tokens routed to expert i (computed from top-1 hard assignments)
     - P_i: average router probability for expert i across the batch
   This penalizes correlation between dispatch frequency and softmax probability.
   Used in Mixtral, MixLoRA, MoE-PEFT.

2. "GShard" auxiliary loss (Lepikhin et al. 2020):
       L_aux = num_experts * Σ_i (m_i / N_total) * P_i
   where m_i is the number of tokens dispatched to expert i.

We adopt formulation #1 (Switch-style), which is what Mixtral / DeepSeekMoE
use and which the user's existing `moe_lora_load_balancing_loss_func` was
modeled after.

Important: The Shared APR Expert is NEVER included in this loss, since by
design it is always active (it should not be load-balanced). The loss is
computed only over routing experts.

Reference numerical scale: typical aux loss coefficient is 0.001 ~ 0.01 in
Mixtral. We follow ``router_aux_loss_coef = 0.01`` from the proposal.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F

from .moe_layer import MoELoRALinear


def _layer_load_balance(router_logits: torch.Tensor, top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Compute Switch-style load-balance loss for one MoE layer.

    Args:
        router_logits: (..., num_experts) - raw router logits.
        top_k_indices: (..., top_k)       - selected expert indices per token.
        num_experts:   number of routing experts.

    Returns:
        Scalar tensor (auxiliary loss for this layer).
    """
    # Flatten leading dims so we can compute per-token statistics.
    router_logits = router_logits.reshape(-1, num_experts)              # (N, E)
    top_k_indices = top_k_indices.reshape(-1, top_k_indices.shape[-1])  # (N, K)

    # P_i: mean softmax probability per expert across all tokens.
    router_probs = F.softmax(router_logits.float(), dim=-1)             # (N, E)
    p_i = router_probs.mean(dim=0)                                      # (E,)

    # f_i: fraction of tokens (counting all top-k slots) routed to expert i.
    one_hot = F.one_hot(top_k_indices, num_classes=num_experts).float() # (N, K, E)
    f_i = one_hot.sum(dim=(0, 1)) / one_hot.sum()                       # (E,)

    # Switch-style: num_experts * Σ_i f_i * P_i
    return num_experts * (f_i * p_i).sum()


def moe_load_balance_loss(model: torch.nn.Module, layers: Optional[Iterable[MoELoRALinear]] = None) -> torch.Tensor:
    """Aggregate load-balance loss across all MoELoRALinear layers in the model.

    Reads ``_cached_router_logits`` and ``_cached_top_k_indices`` set by the
    most recent forward pass. Call this AFTER forward but BEFORE the next forward.

    Args:
        model:  The model containing MoELoRALinear modules.
        layers: Optional explicit iterable of MoELoRALinear layers. If None,
                we auto-discover via ``model.modules()``.

    Returns:
        Scalar tensor (averaged across layers). Returns 0.0 if no layer has
        cached statistics yet.
    """
    if layers is None:
        layers = [m for m in model.modules() if isinstance(m, MoELoRALinear)]
    layers = list(layers)
    if not layers:
        return torch.zeros((), device=next(model.parameters()).device)

    # Under model parallelism, layers live on different GPUs; move each per-layer
    # scalar to one reference device before stacking (torch.stack uses cat, which
    # rejects cross-device tensors).
    ref_device = None
    for layer in layers:
        if layer._cached_router_logits is not None:
            ref_device = layer._cached_router_logits.device
            break

    losses: List[torch.Tensor] = []
    for layer in layers:
        if layer._cached_router_logits is None or layer._cached_top_k_indices is None:
            continue
        l = _layer_load_balance(
            router_logits=layer._cached_router_logits,
            top_k_indices=layer._cached_top_k_indices,
            num_experts=layer.num_routing_experts,
        )
        losses.append(l.to(ref_device) if ref_device is not None else l)

    if not losses:
        return torch.zeros((), device=ref_device or next(model.parameters()).device)
    return torch.stack(losses).mean()
