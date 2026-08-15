"""Patcher: inject MoELoRALinear into a frozen HuggingFace causal LM.

Usage:
    model = AutoModelForCausalLM.from_pretrained(...)
    config = MoEPatchConfig(target_modules=["q_proj","k_proj",...], num_routing_experts=4, top_k=2, ...)
    patch_model_with_moe_lora(model, config)
    model.print_trainable_parameters()  # via report_trainable_parameters

The base model parameters are frozen automatically; only the injected
MoELoRALinear submodules contain trainable LoRA weights + router + gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import os

import torch
import torch.nn as nn

from .moe_layer import MoELoRALinear

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class MoEPatchConfig:
    target_modules: Sequence[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    num_routing_experts: int = 4
    top_k: int = 2
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    use_shared_expert: bool = True
    shared_expert_gate_mode: str = "adaptive"  # "adaptive" | "naive" | "none"
    compute_router_in_fp32: bool = True
    routing_ranks: Optional[Sequence[int]] = None
    routing_alphas: Optional[Sequence[int]] = None
    shared_rank: Optional[int] = None
    shared_alpha: Optional[int] = None
    num_shared_experts: int = 1
    share_routing_A: bool = False  # HydraLoRA baseline: one A per layer, N B heads

    @classmethod
    def from_dict(cls, cfg: dict) -> "MoEPatchConfig":
        """Rebuild from a saved ``patch_config.json``.

        Every eval/generation entry point must go through this: hand-written
        reconstructions have silently dropped fields before (a missing
        num_shared_experts loaded a 3-shared checkpoint as 1-shared and failed
        with a size mismatch), and a dropped field is easy to miss because the
        model still "loads".
        """
        # Bookkeeping written by train_moe_apr.py that is not a layer setting.
        metadata_keys = {"training_mode", "route_by"}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(cfg) - known - metadata_keys
        if unknown:
            raise ValueError(
                f"patch_config.json has fields this MoEPatchConfig does not know: {sorted(unknown)}. "
                "Add them to the dataclass rather than dropping them."
            )
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _get_parent_module(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _is_target_linear(name: str, module: nn.Module, target_modules: Iterable[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    leaf = name.rsplit(".", 1)[-1]
    return leaf in set(target_modules)


def patch_model_with_moe_lora(model: nn.Module, config: MoEPatchConfig) -> List[str]:
    """Replace every target nn.Linear with a MoELoRALinear that wraps it.

    The original Linear is preserved (frozen) inside the new MoELoRALinear.

    Returns:
        List of dotted module names that were replaced.
    """
    # 1) Freeze every base parameter first.
    for p in model.parameters():
        p.requires_grad = False

    # 2) Find all target Linear layers (collect names first to avoid mutating during iter).
    targets: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if _is_target_linear(name, module, config.target_modules):
            targets.append((name, module))

    replaced: List[str] = []
    for name, lin in targets:
        moe_layer = MoELoRALinear(
            base_linear=lin,
            num_routing_experts=config.num_routing_experts,
            top_k=config.top_k,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            use_shared_expert=config.use_shared_expert,
            shared_expert_gate_mode=config.shared_expert_gate_mode,
            compute_router_in_fp32=config.compute_router_in_fp32,
            routing_ranks=config.routing_ranks,
            routing_alphas=config.routing_alphas,
            shared_rank=config.shared_rank,
            shared_alpha=config.shared_alpha,
            num_shared_experts=config.num_shared_experts,
            share_routing_A=config.share_routing_A,
        )
        # Move LoRA experts to base linear's device & dtype.
        moe_layer = moe_layer.to(device=lin.weight.device, dtype=lin.weight.dtype)
        # Router + adaptive_gate must stay fp32 for routing stability (matches
        # Mixtral / DeepSeekMoE practice). We force-upcast them after the
        # general .to() above. Both are tiny so memory cost is negligible.
        if config.compute_router_in_fp32:
            moe_layer.router.to(dtype=torch.float32)
            if moe_layer.adaptive_gate is not None:
                moe_layer.adaptive_gate.to(dtype=torch.float32)
        parent, leaf = _get_parent_module(model, name)
        setattr(parent, leaf, moe_layer)
        replaced.append(name)
    return replaced


def report_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Return (trainable, total, ratio_percent)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct


def collect_moe_layers(model: nn.Module) -> List[MoELoRALinear]:
    return [m for m in model.modules() if isinstance(m, MoELoRALinear)]


def set_inference_bypass(
    model: nn.Module,
    layers: Optional[Iterable[int]] = None,
    projections: Optional[Iterable[str]] = None,
    enable: bool = True,
) -> List[str]:
    """Bypass the MoE-LoRA branch of selected modules at inference time.

    A bypassed module falls back to its frozen base linear, i.e. its experts,
    router and gate are effectively pruned. Used by the RQ3 depth analysis to
    measure what the near-uniform mid-stack modules actually contribute.

    Args:
        model:       a model already patched by :func:`patch_model_with_moe_lora`.
        layers:      transformer layer indices to act on; ``None`` means all.
        projections: projection leaf names (e.g. ``["gate_proj"]``); ``None`` means all.
        enable:      True to bypass, False to restore.

    Returns:
        Names of the modules whose flag was set.
    """
    want_layers = None if layers is None else set(int(x) for x in layers)
    want_proj = None if projections is None else set(projections)
    touched: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, MoELoRALinear):
            continue
        parts = name.split(".")
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        if want_layers is not None and layer_idx not in want_layers:
            continue
        if want_proj is not None and parts[-1] not in want_proj:
            continue
        module.bypass_at_inference = bool(enable)
        touched.append(name)
    return touched


def _parse_layer_spec(spec: str) -> List[int]:
    """``"9-18,24"`` -> ``[9, ..., 18, 24]``."""
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def apply_inference_bypass_from_env(model: nn.Module) -> List[str]:
    """Apply ``MOE_BYPASS_LAYERS`` (and optional ``MOE_BYPASS_PROJ``).

    Mirrors the ``MOE_ABLATE`` convention: evaluation entry points call this
    once after loading the MoE state, so a bypass run needs no code change.
    ``MOE_BYPASS_LAYERS="9-18"`` prunes every adapter module in layers 9..18.
    """
    spec = os.environ.get("MOE_BYPASS_LAYERS", "").strip()
    if not spec:
        return []
    layers = _parse_layer_spec(spec)
    proj_spec = os.environ.get("MOE_BYPASS_PROJ", "").strip()
    projections = [p.strip() for p in proj_spec.split(",") if p.strip()] or None
    touched = set_inference_bypass(model, layers=layers, projections=projections)
    print(f"[moe] inference bypass: layers={spec} proj={projections or 'all'} "
          f"-> {len(touched)} modules pruned", flush=True)
    return touched
