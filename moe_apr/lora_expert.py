"""LoRAExpert: a single low-rank adaptation expert (A B with scaling).

This is the unit-of-expertise used both as a Routing Expert (selected by router)
and as the Shared APR Expert (always active). We keep this class minimal and
framework-agnostic so it can be unit-tested in isolation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRAExpert(nn.Module):
    """Low-rank adapter: y = scaling * B(A(dropout(x))).

    Args:
        in_features:  Input feature dim of the wrapped linear layer.
        out_features: Output feature dim of the wrapped linear layer.
        rank:         LoRA rank.
        alpha:        LoRA alpha; effective scaling = alpha / rank.
        dropout:      Dropout probability applied on the input before A.
        bias:         Whether B has a bias term (default False to match PEFT/LoRA).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )
