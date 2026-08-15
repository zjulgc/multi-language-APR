"""MoELoRALinear: MoE-LoRA layer with Shared APR Expert + Adaptive Gate.

Architecture:

    output(x) = frozen_base_linear(x)                                    # base
              + Σ_i routing_weight_i * RoutingExpert_i(x)                # Top-k routed
              + adaptive_gate * SharedAPRExpert(x)                        # always-active

Joint normalization variant (default; ASE-style):
    Stack [shared_logit | router_logits] and apply a single softmax,
    so that adaptive_gate and routing_weights sum to 1 across the
    union {shared, top_k routing experts}. This empirically prevents
    the shared expert from collapsing to gate=0 or gate=1 (which the
    ASE paper showed happens with naive shared expert designs).

Naive variant (for ablation, paper §4.3):
    Routing weights softmax over top_k; shared expert added with gate=1.0.

We keep the routing implementation token-vectorized: instead of looping per
token over experts, we compute every expert's full output tensor once and
weight-mask it. This is O(num_experts * out_features) per token but allows
batch parallelism. For 4 experts at 7B scale this is acceptable.
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_expert import LoRAExpert

# --------------------------------------------------------------------------- #
#                  Inference-time branch ablation (analysis only)              #
# --------------------------------------------------------------------------- #
#
# The paper claims the always-active shared experts carry the *cross-lingual
# prior* while the routed experts carry the language-specific part. To test that
# claim directly we zero out one branch at inference and re-measure pass@1:
#
#   MOE_ABLATE=none      full model (default; numerically identical to before)
#   MOE_ABLATE=shared    shared-expert contribution -> 0 (routing branch only)
#   MOE_ABLATE=routing   routing-expert contribution -> 0 (shared branch only)
#
# Because the gate is a *joint* softmax over [shared | top-k routing], dropping
# one side leaves weights that no longer sum to 1. Two normalizations:
#
#   MOE_ABLATE_NORM=drop         leave the survivors at their original values
#                                (their sum is < 1, so the adapter is also
#                                globally attenuated)
#   MOE_ABLATE_NORM=drop_renorm  rescale the survivors to sum to 1 -- the
#                                counterfactual "what if this branch had never
#                                existed" (default)
#
# Renormalization only applies to the adaptive (joint-softmax) gate: in "naive"
# mode the shared weights are a constant 1.0 each, not a probability budget, so
# rescaling them would be meaningless and is skipped.
#
# HARD RULE: ablation is a no-op whenever ``self.training`` is True, so no
# training-path number can ever change.

ABLATE_MODES = ("none", "shared", "routing")
ABLATE_NORMS = ("drop", "drop_renorm")

_ABLATE_EPS = 1e-9


class _AblationState:
    __slots__ = ("mode", "norm")

    def __init__(self, mode: str = "none", norm: str = "drop_renorm") -> None:
        self.mode = mode
        self.norm = norm


def _validate_ablation(mode: str, norm: str) -> Tuple[str, str]:
    mode = (mode or "none").strip().lower()
    norm = (norm or "drop_renorm").strip().lower()
    if mode not in ABLATE_MODES:
        raise ValueError(f"MOE_ABLATE must be one of {ABLATE_MODES}, got {mode!r}")
    if norm not in ABLATE_NORMS:
        raise ValueError(f"MOE_ABLATE_NORM must be one of {ABLATE_NORMS}, got {norm!r}")
    return mode, norm


def refresh_moe_ablation_from_env() -> Tuple[str, str]:
    """(Re-)read ``MOE_ABLATE`` / ``MOE_ABLATE_NORM`` into the process state."""
    mode, norm = _validate_ablation(
        os.environ.get("MOE_ABLATE", "none"),
        os.environ.get("MOE_ABLATE_NORM", "drop_renorm"),
    )
    _ABLATION.mode, _ABLATION.norm = mode, norm
    return mode, norm


def set_moe_ablation(mode: Optional[str] = None, norm: Optional[str] = None) -> Tuple[str, str]:
    """Set the ablation programmatically (tests / notebooks)."""
    mode, norm = _validate_ablation(
        _ABLATION.mode if mode is None else mode,
        _ABLATION.norm if norm is None else norm,
    )
    _ABLATION.mode, _ABLATION.norm = mode, norm
    return mode, norm


def get_moe_ablation() -> Tuple[str, str]:
    return _ABLATION.mode, _ABLATION.norm


@contextlib.contextmanager
def moe_ablation(mode: Optional[str] = None, norm: Optional[str] = None):
    """Scoped ablation override, restored on exit."""
    prev = get_moe_ablation()
    try:
        yield set_moe_ablation(mode, norm)
    finally:
        set_moe_ablation(*prev)


_ABLATION = _AblationState()
refresh_moe_ablation_from_env()


class AdaptiveGate(nn.Module):
    """Token-wise gate over a pool of Shared APR Experts.

    Maps hidden state -> num_shared_experts logits per token. These logits are
    then jointly softmax-normalized with router top-k logits, so every shared
    expert weight is bounded in (0, 1) and adapts to whether the token "needs
    each shared APR expert".
    """

    def __init__(self, in_features: int, num_shared_experts: int = 1) -> None:
        super().__init__()
        if num_shared_experts < 1:
            raise ValueError(f"num_shared_experts must be >= 1, got {num_shared_experts}")
        self.num_shared_experts = num_shared_experts
        self.proj = nn.Linear(in_features, num_shared_experts, bias=False)
        nn.init.zeros_(self.proj.weight)  # init at 0 -> uniform softmax over union {shared_*, topk}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., in_features) -> (..., num_shared_experts)
        return self.proj(x)


class MoELoRALinear(nn.Module):
    """Replace a frozen ``nn.Linear`` with MoE-LoRA + Shared APR Expert.

    The wrapped base linear's parameters are frozen; LoRA experts, router, and
    adaptive gate are trainable.

    Args:
        base_linear:                 The original frozen linear layer.
        num_routing_experts:         Number of per-language routing experts.
        top_k:                       How many routing experts to activate per token.
        rank:                        Default LoRA rank for every expert.
        alpha:                       Default LoRA alpha for every expert.
        routing_ranks:               Optional per-routing-expert ranks.
        routing_alphas:              Optional per-routing-expert alphas.
        shared_rank:                 Optional shared-expert rank.
        shared_alpha:                Optional shared-expert alpha.
        num_shared_experts:          Number of always-active shared APR experts (>=1). When > 1,
                                     they are jointly softmax-normalized together with top-k
                                     routing experts, i.e. fine-grained shared expert segmentation
                                     in DeepSeekMoE style.
        dropout:                     Dropout on expert input.
        use_shared_expert:           If False, behave as plain MoE-LoRA (no shared expert). Used
                                     for ablation "MoE-LoRA without Shared".
        shared_expert_gate_mode:     "adaptive" (joint softmax) or "naive" (gate=1.0 each).
                                     "naive" reproduces the ablation "MoE-LoRA + Naive Shared".
        compute_router_in_fp32:      Whether to upcast router logits to fp32 (recommended for stability).
    """

    GATE_MODES = ("adaptive", "naive", "none")

    def __init__(
        self,
        base_linear: nn.Linear,
        num_routing_experts: int = 4,
        top_k: int = 2,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        use_shared_expert: bool = True,
        shared_expert_gate_mode: str = "adaptive",
        compute_router_in_fp32: bool = True,
        routing_ranks: Optional[Sequence[int]] = None,
        routing_alphas: Optional[Sequence[int]] = None,
        shared_rank: Optional[int] = None,
        shared_alpha: Optional[int] = None,
        num_shared_experts: int = 1,
        share_routing_A: bool = False,
    ) -> None:
        super().__init__()
        if shared_expert_gate_mode not in self.GATE_MODES:
            raise ValueError(
                f"shared_expert_gate_mode must be in {self.GATE_MODES}, got {shared_expert_gate_mode}"
            )
        if not use_shared_expert:
            shared_expert_gate_mode = "none"
        if top_k > num_routing_experts:
            raise ValueError(f"top_k({top_k}) must be <= num_routing_experts({num_routing_experts})")
        if routing_ranks is not None and len(routing_ranks) != num_routing_experts:
            raise ValueError(
                f"routing_ranks length({len(routing_ranks)}) must equal num_routing_experts({num_routing_experts})"
            )
        if routing_alphas is not None and len(routing_alphas) != num_routing_experts:
            raise ValueError(
                f"routing_alphas length({len(routing_alphas)}) must equal num_routing_experts({num_routing_experts})"
            )
        if num_shared_experts < 1:
            raise ValueError(f"num_shared_experts must be >= 1, got {num_shared_experts}")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.in_features = in_features
        self.out_features = out_features
        self.num_routing_experts = num_routing_experts
        self.top_k = top_k
        self.rank = rank
        self.alpha = alpha
        self.routing_ranks = list(routing_ranks) if routing_ranks is not None else [rank] * num_routing_experts
        self.routing_alphas = list(routing_alphas) if routing_alphas is not None else [alpha] * num_routing_experts
        self.shared_rank = shared_rank if shared_rank is not None else rank
        self.shared_alpha = shared_alpha if shared_alpha is not None else alpha
        self.num_shared_experts = num_shared_experts if use_shared_expert else 0
        self.use_shared_expert = use_shared_expert
        self.shared_expert_gate_mode = shared_expert_gate_mode
        self.compute_router_in_fp32 = compute_router_in_fp32

        self.routing_experts = nn.ModuleList(
            [
                LoRAExpert(in_features, out_features, expert_rank, expert_alpha, dropout)
                for expert_rank, expert_alpha in zip(self.routing_ranks, self.routing_alphas)
            ]
        )
        # HydraLoRA (Tian et al., NeurIPS 2024) baseline: a single shared
        # down-projection A per layer with N independent B heads, combined by a
        # dense softmax router -- y = W0 x + sum_i w_i B_i (A x). Setting
        # top_k == num_routing_experts makes the router dense, and tying A here
        # gives the asymmetric structure, so no separate layer type is needed.
        self.share_routing_A = share_routing_A
        if share_routing_A:
            if len(set(self.routing_ranks)) != 1:
                raise ValueError(
                    f"share_routing_A needs one rank for all routing experts, got {self.routing_ranks}"
                )
            shared_A = self.routing_experts[0].lora_A
            for expert in self.routing_experts[1:]:
                expert.lora_A = shared_A

        self.router = nn.Linear(in_features, num_routing_experts, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)

        if use_shared_expert:
            self.shared_experts = nn.ModuleList(
                [
                    LoRAExpert(in_features, out_features, self.shared_rank, self.shared_alpha, dropout)
                    for _ in range(num_shared_experts)
                ]
            )
            if shared_expert_gate_mode == "adaptive":
                self.adaptive_gate = AdaptiveGate(in_features, num_shared_experts=num_shared_experts)
            else:
                self.adaptive_gate = None
        else:
            self.shared_experts = nn.ModuleList([])
            self.adaptive_gate = None

        # Cached concatenated expert weights for the inference fast path (see
        # ``_fast_routing_out``). Rebuilt lazily; invalidated by ``train()``.
        self._fast_cat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        # Cached routing stats (for load-balance loss + analysis viz).
        self._cached_router_logits: Optional[torch.Tensor] = None
        self._cached_top_k_weights: Optional[torch.Tensor] = None
        self._cached_top_k_indices: Optional[torch.Tensor] = None
        self._cached_shared_weight: Optional[torch.Tensor] = None  # (..., num_shared_experts)

    @property
    def shared_expert(self) -> Optional["LoRAExpert"]:
        """Back-compat alias: returns the lone shared expert when num_shared_experts==1.

        Returns ``None`` when shared expert is disabled or when there is more than
        one shared expert (in that case use ``self.shared_experts`` directly).
        """
        if not self.use_shared_expert:
            return None
        if len(self.shared_experts) == 1:
            return self.shared_experts[0]
        return None

    # ------------------------------------------------------------------ #
    #                              Forward                                #
    # ------------------------------------------------------------------ #

    def _compute_routing_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute router logits, top-k weights, top-k indices, and (optional) shared gate weights.

        Returns:
            router_logits:  (..., num_routing_experts) - full router logits (for load-balance loss)
            top_k_weights:  (..., top_k) - normalized routing weights
            top_k_indices:  (..., top_k) - selected expert indices
            shared_weight:  (..., num_shared_experts) or None - normalized per-shared-expert weights
        """
        router_input = x.float() if self.compute_router_in_fp32 else x
        router_logits = self.router(router_input)  # (..., num_routing_experts)

        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)

        if self.use_shared_expert and self.shared_expert_gate_mode == "adaptive":
            shared_logits = self.adaptive_gate(router_input)  # (..., num_shared_experts)
            S = self.num_shared_experts
            combined = torch.cat([shared_logits, top_k_logits], dim=-1)  # (..., S + top_k)
            combined_weights = F.softmax(combined, dim=-1)
            shared_weight = combined_weights[..., :S]
            top_k_weights = combined_weights[..., S:]
        else:
            top_k_weights = F.softmax(top_k_logits, dim=-1)
            if self.use_shared_expert and self.shared_expert_gate_mode == "naive":
                # Each shared expert contributes with weight 1.0 (broadcast).
                shared_weight = top_k_weights.new_ones(
                    *top_k_weights.shape[:-1], self.num_shared_experts
                )
            else:
                shared_weight = None

        top_k_weights, shared_weight = self._apply_branch_ablation(top_k_weights, shared_weight)

        # Cast back to compute dtype.
        top_k_weights = top_k_weights.to(x.dtype)
        if shared_weight is not None:
            shared_weight = shared_weight.to(x.dtype)

        return router_logits, top_k_weights, top_k_indices, shared_weight

    def _apply_branch_ablation(
        self, top_k_weights: torch.Tensor, shared_weight: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Zero one branch's gate weights (inference-only; see module docstring).

        Both forward paths -- the concatenated fast path and the per-expert loop
        -- consume exactly these two tensors, so acting here keeps them
        automatically consistent: a zero weight makes the expert contribute
        exactly 0 in either formulation.

        Returns the tensors unchanged in training mode or when the ablation is
        ``none``, so the default numerical behaviour is bit-identical.
        """
        mode = _ABLATION.mode
        if mode == "none" or self.training:
            return top_k_weights, shared_weight

        # Renormalizing only makes sense for the joint softmax; "naive" shared
        # gates are constants, and "none" has no shared branch at all.
        renorm = _ABLATION.norm == "drop_renorm" and self.shared_expert_gate_mode == "adaptive"

        if mode == "shared":
            if shared_weight is None:
                return top_k_weights, shared_weight  # no shared branch to remove
            if renorm:
                denom = top_k_weights.sum(dim=-1, keepdim=True).clamp_min(_ABLATE_EPS)
                top_k_weights = top_k_weights / denom
            shared_weight = torch.zeros_like(shared_weight)
        elif mode == "routing":
            if shared_weight is not None and renorm:
                denom = shared_weight.sum(dim=-1, keepdim=True).clamp_min(_ABLATE_EPS)
                shared_weight = shared_weight / denom
            top_k_weights = torch.zeros_like(top_k_weights)

        return top_k_weights, shared_weight

    # ------------------------------------------------------------------ #
    #                       Inference fast path                           #
    # ------------------------------------------------------------------ #

    def _fast_path_available(self) -> bool:
        """Whether the concatenated-expert form may replace the per-expert loop.

        Never in training: the loop draws an independent dropout mask per expert,
        which the concatenated form (one shared input) cannot reproduce. Set
        ``MOE_FAST_INFER=0`` to force the loop at inference too (used by the
        equivalence test in ``tests/``).
        """
        if self.training or os.environ.get("MOE_FAST_INFER", "1") == "0":
            return False
        return all(e.lora_B.bias is None for e in self.routing_experts)

    def _fast_experts(self) -> list:
        """Routing experts followed by the always-active shared experts.

        Shared experts are folded into the same two matmuls: they are just
        experts whose gate weight is never zero, so concatenating them saves
        another 2 matmuls per shared expert per patched projection.
        """
        return list(self.routing_experts) + list(self.shared_experts)

    def _build_fast_cat(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experts = self._fast_experts()
        A = torch.cat([e.lora_A.weight for e in experts], dim=0)  # (R, in)
        B = torch.cat([e.lora_B.weight for e in experts], dim=1)  # (out, R)
        dev = A.device
        col_expert = torch.cat(
            [torch.full((e.rank,), i, dtype=torch.long, device=dev)
             for i, e in enumerate(experts)]
        )  # (R,) which expert owns each low-rank column
        col_scaling = torch.cat(
            [torch.full((e.rank,), e.scaling, dtype=A.dtype, device=dev)
             for e in experts]
        )  # (R,) per-expert alpha/rank, applied in the rank space
        return A, B, col_expert, col_scaling

    def _get_fast_cat(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.routing_experts[0].lora_A.weight
        cache = self._fast_cat
        if cache is None or cache[0].device != ref.device or cache[0].dtype != ref.dtype:
            cache = self._build_fast_cat()
            self._fast_cat = cache
        return cache

    def _fast_routing_out(self, flat_x: torch.Tensor, flat_w: torch.Tensor) -> torch.Tensor:
        """Weighted sum over routing + shared experts as two dense matmuls.

        ``flat_x``: (T, in_features); ``flat_w``: (T, num_routing_experts [+
        num_shared_experts]) with zeros for unselected routing experts -- those
        contribute exactly 0, so the result equals the loop's weighted sum (up
        to float summation order).
        """
        A, B, col_expert, col_scaling = self._get_fast_cat()
        h = F.linear(flat_x, A)                                     # (T, R)
        w_cols = flat_w.index_select(1, col_expert) * col_scaling   # (T, R)
        return F.linear(h * w_cols.to(h.dtype), B)                  # (T, out)

    def train(self, mode: bool = True) -> "MoELoRALinear":
        # Weights may change while training -> drop the concatenated cache.
        self._fast_cat = None
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        base_out = self.base(x)

        router_logits, top_k_weights, top_k_indices, shared_weight = self._compute_routing_weights(x)

        # ------ Routing experts --------------------------------------------
        # Only run an expert on tokens routed to it. The earlier dense variant
        # computed every routing expert on the full batch and then masked
        # outputs, which is simple but roughly E/K times more expensive for
        # top-k routing. With E=4 and K=2 this cuts routing-expert matmul work
        # by about half while preserving the same weighted sum semantics.
        topk_one_hot = F.one_hot(top_k_indices, num_classes=self.num_routing_experts).to(top_k_weights.dtype)
        weight_per_expert = (top_k_weights.unsqueeze(-1) * topk_one_hot).sum(dim=-2)  # (..., num_routing_experts)

        flat_x = x.reshape(-1, self.in_features)
        flat_weight_per_expert = weight_per_expert.reshape(-1, self.num_routing_experts)

        fast = self._fast_path_available()
        if fast and self.shared_experts and shared_weight is None:
            fast = False  # gate mode "none": shared experts have no weights to fold in

        if fast:
            # Two dense matmuls over the concatenated experts (routing + shared).
            # The loop below skips unselected experts, but paying `torch.any` per
            # expert costs a device sync -- with ~196 patched projections x E
            # experts that dominates autoregressive decode (measured ~30-90x
            # slower than base on long xCodeEval generations). Zero-weight
            # experts contribute exactly zero here, so the sum is unchanged.
            all_w = flat_weight_per_expert
            if self.shared_experts:
                all_w = torch.cat(
                    [all_w, shared_weight.reshape(-1, self.num_shared_experts).to(all_w.dtype)], dim=1
                )
            moe_out = self._fast_routing_out(flat_x, all_w).reshape(base_out.shape)
            if moe_out.dtype != base_out.dtype:
                moe_out = moe_out.to(base_out.dtype)
            self._cached_router_logits = router_logits
            self._cached_top_k_weights = top_k_weights.detach()
            self._cached_top_k_indices = top_k_indices.detach()
            self._cached_shared_weight = shared_weight.detach() if shared_weight is not None else None
            return base_out + moe_out

        moe_out = torch.zeros_like(base_out)
        flat_moe_out = moe_out.reshape(-1, self.out_features)
        for expert_idx, expert in enumerate(self.routing_experts):
            w = flat_weight_per_expert[:, expert_idx]
            selected = w.detach() != 0
            if not torch.any(selected):
                continue
            token_idx = selected.nonzero(as_tuple=False).squeeze(-1)
            expert_out = expert(flat_x.index_select(0, token_idx))
            weighted = w.index_select(0, token_idx).unsqueeze(-1) * expert_out
            # index_add_ requires self and source to share dtype. Under HF Trainer's
            # bf16 autocast the LoRA / weight path may temporarily land in fp32, so
            # force a cast back to the accumulator dtype before the in-place add.
            if weighted.dtype != flat_moe_out.dtype:
                weighted = weighted.to(flat_moe_out.dtype)
            flat_moe_out.index_add_(0, token_idx, weighted)

        # ------ Shared experts ---------------------------------------------
        if self.use_shared_expert and shared_weight is not None:
            for s_idx, expert in enumerate(self.shared_experts):
                moe_out = moe_out + shared_weight[..., s_idx : s_idx + 1] * expert(x)

        # Cache for load-balance loss + analysis.
        self._cached_router_logits = router_logits
        self._cached_top_k_weights = top_k_weights.detach()
        self._cached_top_k_indices = top_k_indices.detach()
        self._cached_shared_weight = shared_weight.detach() if shared_weight is not None else None

        return base_out + moe_out

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"num_routing_experts={self.num_routing_experts}, top_k={self.top_k}, "
            f"rank={self.rank}, alpha={self.alpha}, routing_ranks={self.routing_ranks}, "
            f"shared_rank={self.shared_rank}, num_shared_experts={self.num_shared_experts}, "
            f"shared={self.use_shared_expert}, gate={self.shared_expert_gate_mode}, "
            f"share_routing_A={self.share_routing_A}"
        )
