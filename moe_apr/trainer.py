"""Trainer for single-stage MoE-LoRA + Shared APR Expert training."""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import Sampler

try:
    from transformers import Trainer
except Exception:  # pragma: no cover
    Trainer = object  # type: ignore

from .load_balance import moe_load_balance_loss
from .moe_metrics import moe_routing_stats
from .model_patcher import collect_moe_layers


class MoETrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        router_aux_loss_coef: float = 0.01,
        train_sampler: Optional[Sampler] = None,
        log_moe_metrics: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.router_aux_loss_coef = router_aux_loss_coef
        self._train_sampler_override = train_sampler
        self.log_moe_metrics = log_moe_metrics
        self._last_aux_loss = 0.0
        self._moe_metric_accum: Dict[str, float] = {}
        self._moe_metric_steps = 0
        # MoE routing stats do many .item() GPU->CPU syncs per layer. Computing
        # them on every micro-step (grad_accum times per optimizer step) is a
        # sync storm that dominates step time for no benefit -- the values are
        # only averaged and logged every logging_steps. Sample them once per
        # optimizer step instead (statistically identical for logging).
        self._micro_step = 0

        self._moe_layers = collect_moe_layers(self.model)
        if not self._moe_layers:
            raise RuntimeError("MoETrainer needs MoELoRALinear layers; did you patch the model?")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        route_ids = inputs.pop("route_id", None)
        labels = inputs.get("labels", None)

        outputs = model(**inputs)
        lm_loss = outputs.loss
        loss = lm_loss

        if self.router_aux_loss_coef > 0:
            aux_loss = moe_load_balance_loss(model, layers=self._moe_layers)
            loss = loss + self.router_aux_loss_coef * aux_loss
            self._last_aux_loss = aux_loss.detach().float().item() if aux_loss.numel() > 0 else 0.0
        else:
            self._last_aux_loss = 0.0

        if self.log_moe_metrics:
            self._micro_step += 1
            stride = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
            # Only sample stats once per optimizer step (last micro-step of the
            # accumulation window) to avoid the per-micro-step .item() sync storm.
            if self._micro_step % stride == 0:
                label_mask = (labels != -100) if labels is not None else None
                for key, val in moe_routing_stats(
                    self._moe_layers,
                    route_ids=route_ids,
                    label_mask=label_mask,
                ).items():
                    self._moe_metric_accum[key] = self._moe_metric_accum.get(key, 0.0) + val
                self._moe_metric_steps += 1

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[Sampler]:  # type: ignore[override]
        if self._train_sampler_override is not None:
            return self._train_sampler_override
        return super()._get_train_sampler()

    def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if self.router_aux_loss_coef > 0:
            logs = {**logs, "aux_loss": self._last_aux_loss}
        if self.log_moe_metrics and self._moe_metric_steps > 0:
            n = float(self._moe_metric_steps)
            for key, total in self._moe_metric_accum.items():
                logs[f"moe/{key}"] = total / n
            self._moe_metric_accum.clear()
            self._moe_metric_steps = 0
        super().log(logs, *args, **kwargs)
