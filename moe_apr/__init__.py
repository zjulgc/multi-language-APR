"""MoE-LoRA with Semantic Shared APR Expert for Multilingual Program Repair.

Module layout:
    lora_expert.py        - LoRAExpert: single LoRA expert unit.
    moe_layer.py          - MoELoRALinear: full MoE layer wrapping a frozen base nn.Linear.
                            Supports Shared APR Expert + Adaptive Gate + Top-k routing.
    load_balance.py       - Load balancing loss (only over routing experts; shared expert excluded).
    trainer.py            - Trainer extension for single-stage MoE training.
    data_utils.py         - BalancedLanguageSampler for balanced per-language sampling.
    model_patcher.py      - Helper to inject MoELoRALinear into an existing HF causal LM.
"""

from .lora_expert import LoRAExpert
from .moe_layer import MoELoRALinear, AdaptiveGate
from .load_balance import moe_load_balance_loss
from .moe_metrics import moe_routing_stats

__all__ = [
    "LoRAExpert",
    "MoELoRALinear",
    "AdaptiveGate",
    "moe_load_balance_loss",
    "moe_routing_stats",
]
