"""Training-time MoE routing statistics (for SwanLab / Trainer logs).

Reads per-layer caches populated by ``MoELoRALinear.forward`` and returns
scalar summaries averaged across MoE layers. Intended to be called once per
training step inside ``MoETrainer.compute_loss``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from .moe_layer import MoELoRALinear


def _expert_fraction(top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Fraction of top-k routing slots assigned to each expert (Switch-style f_i)."""
    flat = top_k_indices.reshape(-1, top_k_indices.shape[-1])
    one_hot = F.one_hot(flat, num_classes=num_experts).float()
    return one_hot.sum(dim=(0, 1)) / one_hot.sum().clamp(min=1.0)


def _router_entropy(router_logits: torch.Tensor, num_experts: int) -> float:
    """Mean softmax entropy of the router distribution, averaged over tokens."""
    probs = F.softmax(router_logits.reshape(-1, num_experts).float(), dim=-1)
    ent = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
    return float(ent.item())


def _route_purity(
    top_k_indices: torch.Tensor,
    route_ids: torch.Tensor,
    label_mask: Optional[torch.Tensor],
) -> float:
    """Fraction of response tokens whose top-1 routed expert matches route_id."""
    top1 = top_k_indices[..., 0]
    b, t = top1.shape
    target = route_ids.to(top1.device).long().view(b, 1).expand(b, t)
    if label_mask is not None:
        valid = label_mask.to(top1.device).bool()
    else:
        valid = torch.ones_like(top1, dtype=torch.bool)
    denom = valid.float().sum().clamp(min=1.0)
    return float(((top1 == target) & valid).float().sum().item() / denom.item())


def _route_nmi(
    top_k_indices: torch.Tensor,
    route_ids: torch.Tensor,
    label_mask: Optional[torch.Tensor],
    num_experts: int,
) -> float:
    """Normalized mutual information between the route target (language / bug-type id)
    and the top-1 selected expert, over valid (response) tokens.

    Unlike ``route_purity``, NMI is ASSIGNMENT-INVARIANT: it credits the router
    for specialization even when a target maps to a non-canonical expert index
    (e.g. the free router consistently sends Python to expert 3 rather than to
    Python's canonical id 5). NMI in [0, 1]: ~0 = no specialization (routing is
    independent of language), 1 = each language deterministically owns an expert.
    """
    top1 = top_k_indices[..., 0]
    b, t = top1.shape
    target = route_ids.to(top1.device).long().view(b, 1).expand(b, t)
    valid = label_mask.to(top1.device).bool() if label_mask is not None \
        else torch.ones_like(top1, dtype=torch.bool)
    e = top1[valid].reshape(-1).long()
    g = target[valid].reshape(-1).long()
    if e.numel() == 0:
        return 0.0
    G = int(g.max().item()) + 1
    joint = torch.bincount(g * num_experts + e, minlength=G * num_experts).float()
    p = (joint / joint.sum()).reshape(G, num_experts)
    pg = p.sum(dim=1, keepdim=True)  # (G,1) target marginal
    pe = p.sum(dim=0, keepdim=True)  # (1,E) expert marginal
    nz = p > 0
    mi = (p[nz] * (p[nz].log()
                   - pg.expand_as(p)[nz].log()
                   - pe.expand_as(p)[nz].log())).sum()
    hg = -(pg[pg > 0] * pg[pg > 0].log()).sum()
    he = -(pe[pe > 0] * pe[pe > 0].log()).sum()
    denom = torch.sqrt((hg * he).clamp(min=1e-12))
    return float((mi / denom).clamp(0.0, 1.0).item())


def moe_routing_stats(
    layers: Iterable[MoELoRALinear],
    *,
    route_ids: Optional[torch.Tensor] = None,
    label_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Aggregate routing statistics across MoE layers for the last forward pass.

    Returns keys such as ``router_entropy``, ``load_imbalance``, ``shared_gate``,
    ``route_purity`` (when ``route_ids`` given), and ``expert_util_{i}``.
    """
    layers = list(layers)
    if not layers:
        return {}

    util_sums: Optional[torch.Tensor] = None
    entropies: List[float] = []
    imbalances: List[float] = []
    shared_gates: List[float] = []
    purities: List[float] = []
    nmis: List[float] = []
    num_experts: Optional[int] = None

    for layer in layers:
        if layer._cached_router_logits is None or layer._cached_top_k_indices is None:
            continue
        e = layer.num_routing_experts
        num_experts = e
        f_i = _expert_fraction(layer._cached_top_k_indices, e)
        util_sums = f_i if util_sums is None else util_sums + f_i
        entropies.append(_router_entropy(layer._cached_router_logits, e))
        imbalances.append(float((f_i.max() - f_i.min()).item()))
        if layer._cached_shared_weight is not None:
            # Total mass on shared experts per token, then batch mean.
            shared_gates.append(float(layer._cached_shared_weight.float().sum(dim=-1).mean().item()))
        if route_ids is not None:
            purities.append(
                _route_purity(layer._cached_top_k_indices, route_ids, label_mask)
            )
            nmis.append(
                _route_nmi(layer._cached_top_k_indices, route_ids, label_mask, e)
            )

    if util_sums is None or num_experts is None:
        return {}

    util_avg = (util_sums / max(len(entropies), 1)).tolist()
    out: Dict[str, float] = {
        "router_entropy": sum(entropies) / len(entropies),
        "load_imbalance": sum(imbalances) / len(imbalances),
    }
    if shared_gates:
        out["shared_gate"] = sum(shared_gates) / len(shared_gates)
    if purities:
        out["route_purity"] = sum(purities) / len(purities)
        # ~1/E under uniform routing; useful as a quick sanity reference.
        out["route_purity_random"] = 1.0 / num_experts
    if nmis:
        # Assignment-invariant specialization signal; ~0 = router independent of
        # language (reproduces the paper's §5.3 null), rises as experts specialize.
        out["route_nmi"] = sum(nmis) / len(nmis)
    for i, u in enumerate(util_avg):
        out[f"expert_util_{i}"] = u
    return out
