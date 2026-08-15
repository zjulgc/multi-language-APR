"""Unit tests for moe_apr.

Run: ``python -m pytest moe_apr/tests/ -v``
or:  ``python -m unittest moe_apr.tests.test_moe_layer``

These tests run on CPU with tiny dimensions so they finish in <2s.
"""

from __future__ import annotations

import math
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_apr import LoRAExpert, MoELoRALinear, AdaptiveGate
from moe_apr.load_balance import moe_load_balance_loss
from moe_apr.moe_layer import get_moe_ablation, moe_ablation, set_moe_ablation
from moe_apr.model_patcher import (
    MoEPatchConfig,
    collect_moe_layers,
    patch_model_with_moe_lora,
)


class TestLoRAExpert(unittest.TestCase):
    def test_zero_init_yields_zero_output(self):
        torch.manual_seed(0)
        expert = LoRAExpert(in_features=8, out_features=16, rank=4, alpha=8, dropout=0.0)
        x = torch.randn(2, 3, 8)
        y = expert(x)
        # Because lora_B is zero-init, output must start at zero.
        self.assertTrue(torch.allclose(y, torch.zeros_like(y)))

    def test_after_one_step_nonzero(self):
        torch.manual_seed(0)
        expert = LoRAExpert(in_features=8, out_features=16, rank=4, alpha=8, dropout=0.0)
        x = torch.randn(2, 3, 8)
        target = torch.randn(2, 3, 16)
        opt = torch.optim.SGD(expert.parameters(), lr=0.1)
        y = expert(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        opt.step()
        # B has now received non-zero gradient.
        self.assertFalse(torch.allclose(expert.lora_B.weight, torch.zeros_like(expert.lora_B.weight)))


class TestMoELoRALinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.in_features = 16
        self.out_features = 32
        self.base = nn.Linear(self.in_features, self.out_features, bias=True)

    def _build(self, **kw):
        kwargs = dict(
            num_routing_experts=4,
            top_k=2,
            rank=4,
            alpha=8,
            dropout=0.0,
            use_shared_expert=True,
            shared_expert_gate_mode="adaptive",
        )
        kwargs.update(kw)
        return MoELoRALinear(self.base, **kwargs)

    def test_forward_initially_equals_base(self):
        """Right after init (every LoRA B is zero), forward must equal base."""
        layer = self._build()
        x = torch.randn(2, 5, self.in_features)
        y_base = self.base(x)
        y = layer(x)
        self.assertTrue(torch.allclose(y, y_base, atol=1e-6))

    def test_base_frozen(self):
        layer = self._build()
        for p in layer.base.parameters():
            self.assertFalse(p.requires_grad)
        # Trainable params: 4 routing + 1 shared + router + adaptive_gate.
        trainable_modules = [n for n, p in layer.named_parameters() if p.requires_grad]
        self.assertTrue(any("routing_experts" in n for n in trainable_modules))
        self.assertTrue(any("shared_expert" in n for n in trainable_modules))
        self.assertTrue(any("router" in n for n in trainable_modules))
        self.assertTrue(any("adaptive_gate" in n for n in trainable_modules))

    def test_backward_flows_to_router_and_gate(self):
        layer = self._build()
        # Force B to be non-zero so gradients reach router/gate via routing weights.
        with torch.no_grad():
            for e in layer.routing_experts:
                e.lora_B.weight.normal_(0, 0.01)
            layer.shared_expert.lora_B.weight.normal_(0, 0.01)
        x = torch.randn(2, 5, self.in_features)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(layer.router.weight.grad)
        self.assertTrue(torch.any(layer.router.weight.grad != 0))
        self.assertIsNotNone(layer.adaptive_gate.proj.weight.grad)
        self.assertTrue(torch.any(layer.adaptive_gate.proj.weight.grad != 0))

    def test_no_shared_expert_mode(self):
        layer = self._build(use_shared_expert=False)
        self.assertIsNone(layer.shared_expert)
        self.assertIsNone(layer.adaptive_gate)
        x = torch.randn(2, 5, self.in_features)
        y = layer(x)
        # Should still match base initially.
        y_base = self.base(x)
        self.assertTrue(torch.allclose(y, y_base, atol=1e-6))

    def test_naive_shared_expert_gate(self):
        layer = self._build(shared_expert_gate_mode="naive")
        # Make shared expert non-zero so we can detect the gate=1 contribution.
        with torch.no_grad():
            layer.shared_expert.lora_B.weight.fill_(0.1)
        x = torch.randn(2, 5, self.in_features)
        # Shared expert weight must equal 1.0 in naive mode.
        layer(x)
        self.assertIsNotNone(layer._cached_shared_weight)
        self.assertTrue(torch.allclose(
            layer._cached_shared_weight, torch.ones_like(layer._cached_shared_weight),
        ))

    def test_routing_weights_sum_to_one_with_shared(self):
        """In adaptive mode, sum(shared_weights) + sum(top_k_weights) == 1 per token."""
        layer = self._build()
        x = torch.randn(2, 5, self.in_features)
        layer(x)
        total = layer._cached_shared_weight.sum(dim=-1) + layer._cached_top_k_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))

    def test_multi_shared_experts_initially_match_base(self):
        layer = self._build(num_shared_experts=2)
        self.assertEqual(len(layer.shared_experts), 2)
        # When num_shared > 1, the back-compat alias must return None.
        self.assertIsNone(layer.shared_expert)
        x = torch.randn(2, 5, self.in_features)
        y = layer(x)
        # Zero-init B over all experts => output equals base.
        self.assertTrue(torch.allclose(y, self.base(x), atol=1e-6))
        # cached shared weight has shape (..., num_shared_experts).
        self.assertEqual(layer._cached_shared_weight.shape[-1], 2)

    def test_multi_shared_experts_weights_sum_to_one(self):
        layer = self._build(num_shared_experts=3, top_k=2)
        x = torch.randn(2, 5, self.in_features)
        layer(x)
        total = layer._cached_shared_weight.sum(dim=-1) + layer._cached_top_k_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))

    def test_multi_shared_experts_naive_mode(self):
        layer = self._build(num_shared_experts=2, shared_expert_gate_mode="naive")
        x = torch.randn(2, 5, self.in_features)
        layer(x)
        # In naive mode each shared expert contributes with weight 1.0.
        self.assertEqual(layer._cached_shared_weight.shape[-1], 2)
        self.assertTrue(torch.allclose(
            layer._cached_shared_weight, torch.ones_like(layer._cached_shared_weight),
        ))

    def test_multi_shared_experts_backward_reaches_all(self):
        layer = self._build(num_shared_experts=2)
        # Make all experts' B non-zero so routing/gate gradients flow.
        with torch.no_grad():
            for e in layer.routing_experts:
                e.lora_B.weight.normal_(0, 0.01)
            for e in layer.shared_experts:
                e.lora_B.weight.normal_(0, 0.01)
        x = torch.randn(2, 5, self.in_features)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        for e in layer.shared_experts:
            self.assertIsNotNone(e.lora_A.weight.grad)
            self.assertTrue(torch.any(e.lora_A.weight.grad != 0))
        # AdaptiveGate now projects to num_shared_experts dims; gradient should be non-zero.
        self.assertEqual(layer.adaptive_gate.proj.weight.shape, torch.Size([2, self.in_features]))
        self.assertIsNotNone(layer.adaptive_gate.proj.weight.grad)
        self.assertTrue(torch.any(layer.adaptive_gate.proj.weight.grad != 0))

    def test_routing_weights_sum_to_one_no_shared(self):
        """Without shared expert: top_k weights sum to 1."""
        layer = self._build(use_shared_expert=False)
        x = torch.randn(2, 5, self.in_features)
        layer(x)
        total = layer._cached_top_k_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))

    def test_top_k_indices_distinct(self):
        layer = self._build()
        x = torch.randn(2, 5, self.in_features)
        layer(x)
        ind = layer._cached_top_k_indices
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                row = ind[i, j].tolist()
                self.assertEqual(len(set(row)), len(row), f"Duplicate experts: {row}")


class TestLoadBalance(unittest.TestCase):
    def test_load_balance_loss_is_positive(self):
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = MoELoRALinear(base, num_routing_experts=4, top_k=2, rank=4, alpha=8, dropout=0.0)
        x = torch.randn(2, 8, 16)
        layer(x)
        # Balanced: load loss should be ~ num_experts * (1/num_experts)^2 * num_experts = 1.
        loss = moe_load_balance_loss(nn.ModuleList([layer]))
        self.assertTrue(loss.item() > 0)
        # With top-k=2 over 4 experts, in the perfectly balanced case the loss should be
        # close to top_k / num_experts = 0.5; allow a wide tolerance for tiny batches.
        self.assertTrue(0.1 < loss.item() < 4.0, f"Got loss={loss.item():.4f}")

    def test_load_balance_zero_when_no_cache(self):
        base = nn.Linear(16, 32)
        layer = MoELoRALinear(base, num_routing_experts=4, top_k=2, rank=4, alpha=8, dropout=0.0)
        loss = moe_load_balance_loss(nn.ModuleList([layer]))
        self.assertEqual(loss.item(), 0.0)


class TestModelPatcher(unittest.TestCase):
    def _toy_model(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16, bias=False)
                self.k_proj = nn.Linear(16, 16, bias=False)
                self.v_proj = nn.Linear(16, 16, bias=False)
                self.o_proj = nn.Linear(16, 16, bias=False)
                self.norm = nn.LayerNorm(16)

            def forward(self, x):
                return self.o_proj(self.q_proj(self.norm(x)))

        return Toy()

    def test_patch_replaces_target_modules(self):
        torch.manual_seed(0)
        model = self._toy_model()
        config = MoEPatchConfig(
            target_modules=["q_proj", "v_proj"],
            num_routing_experts=4,
            top_k=2,
            rank=4,
            alpha=8,
        )
        replaced = patch_model_with_moe_lora(model, config)
        self.assertEqual(set(replaced), {"q_proj", "v_proj"})
        self.assertIsInstance(model.q_proj, MoELoRALinear)
        self.assertIsInstance(model.v_proj, MoELoRALinear)
        self.assertIsInstance(model.k_proj, nn.Linear)  # not replaced
        # Original linear is preserved as `.base`
        self.assertEqual(model.q_proj.base.in_features, 16)

    def test_patch_freezes_base(self):
        model = self._toy_model()
        config = MoEPatchConfig(target_modules=["q_proj", "v_proj"], rank=4, alpha=8)
        patch_model_with_moe_lora(model, config)
        # Base linear weights are frozen.
        self.assertFalse(model.q_proj.base.weight.requires_grad)
        # Untouched layer is also frozen (we freeze everything by default).
        self.assertFalse(model.k_proj.weight.requires_grad)

    def test_collect_moe_layers(self):
        model = self._toy_model()
        config = MoEPatchConfig(target_modules=["q_proj", "v_proj"], rank=4, alpha=8)
        patch_model_with_moe_lora(model, config)
        layers = collect_moe_layers(model)
        self.assertEqual(len(layers), 2)

    def test_patch_propagates_num_shared_experts(self):
        model = self._toy_model()
        config = MoEPatchConfig(
            target_modules=["q_proj", "v_proj"], rank=4, alpha=8, num_shared_experts=2,
        )
        patch_model_with_moe_lora(model, config)
        for m in collect_moe_layers(model):
            self.assertEqual(len(m.shared_experts), 2)
            self.assertEqual(m.adaptive_gate.proj.weight.shape[0], 2)


class TestLoadStateBackcompat(unittest.TestCase):
    """Backward compat: old checkpoints used ``shared_expert.*`` instead of
    ``shared_experts.0.*``. ``load_moe_state`` must remap on the fly so legacy
    checkpoints continue to load against the new module layout."""

    def test_legacy_shared_expert_key_remap(self):
        from train_moe_apr import load_moe_state
        torch.manual_seed(0)
        base = nn.Linear(16, 32, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=4, top_k=2, rank=4, alpha=8, dropout=0.0)

        # Build a fake legacy state dict for the shared expert (single).
        legacy_sd = {
            "shared_expert.lora_A.weight": torch.full_like(layer.shared_experts[0].lora_A.weight, 0.123),
            "shared_expert.lora_B.weight": torch.full_like(layer.shared_experts[0].lora_B.weight, 0.456),
        }
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(legacy_sd, tmp.name)
            load_moe_state(layer, tmp.name, strict=False)

        self.assertTrue(torch.allclose(
            layer.shared_experts[0].lora_A.weight,
            torch.full_like(layer.shared_experts[0].lora_A.weight, 0.123),
        ))
        self.assertTrue(torch.allclose(
            layer.shared_experts[0].lora_B.weight,
            torch.full_like(layer.shared_experts[0].lora_B.weight, 0.456),
        ))


class TestInferenceFastPath(unittest.TestCase):
    """The concatenated-expert inference path must match the per-expert loop.

    ``MOE_FAST_INFER=0`` forces the loop, so the two are compared on the same
    layer with the same inputs.
    """

    @staticmethod
    def _randomize(layer: MoELoRALinear) -> None:
        # lora_B is zero-init by design, which would make every path trivially
        # equal -- give every expert a distinct non-zero B.
        for e in list(layer.routing_experts) + list(layer.shared_experts):
            torch.nn.init.normal_(e.lora_A.weight, std=0.05)
            torch.nn.init.normal_(e.lora_B.weight, std=0.05)

    @staticmethod
    def _both(layer: MoELoRALinear, x: torch.Tensor):
        import os
        layer.eval()
        with torch.no_grad():
            os.environ["MOE_FAST_INFER"] = "0"
            ref = layer(x).clone()
            os.environ["MOE_FAST_INFER"] = "1"
            fast = layer(x).clone()
        os.environ.pop("MOE_FAST_INFER", None)
        return ref, fast

    def test_matches_loop_topk1_with_shared(self):
        """s3-shaped config: 11 routing experts, top-1, 3 shared experts."""
        torch.manual_seed(0)
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=11, top_k=1, rank=4, alpha=8,
                              dropout=0.0, num_shared_experts=3, shared_rank=4)
        self._randomize(layer)
        x = torch.randn(2, 7, 32)
        ref, fast = self._both(layer, x)
        self.assertTrue(torch.allclose(ref, fast, atol=1e-6, rtol=1e-5),
                        f"max abs diff {(ref - fast).abs().max().item():.3e}")

    def test_matches_loop_topk4_no_shared(self):
        """A2-shaped config: 14 routing experts, top-4, no shared expert."""
        torch.manual_seed(1)
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=14, top_k=4, rank=4, alpha=8,
                              dropout=0.0, use_shared_expert=False)
        self._randomize(layer)
        x = torch.randn(3, 5, 32)
        ref, fast = self._both(layer, x)
        self.assertTrue(torch.allclose(ref, fast, atol=1e-6, rtol=1e-5),
                        f"max abs diff {(ref - fast).abs().max().item():.3e}")

    def test_matches_loop_heterogeneous_ranks(self):
        torch.manual_seed(2)
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=4, top_k=2, rank=4, alpha=8,
                              dropout=0.0, routing_ranks=[2, 4, 8, 4],
                              routing_alphas=[4, 8, 16, 8])
        self._randomize(layer)
        x = torch.randn(2, 6, 32)
        ref, fast = self._both(layer, x)
        self.assertTrue(torch.allclose(ref, fast, atol=1e-6, rtol=1e-5),
                        f"max abs diff {(ref - fast).abs().max().item():.3e}")

    def test_single_token_decode_shape(self):
        """Autoregressive decode step: batch=1, seq=1 -- the case the loop was slow on."""
        torch.manual_seed(3)
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=11, top_k=1, rank=4, alpha=8,
                              dropout=0.0, num_shared_experts=3, shared_rank=4)
        self._randomize(layer)
        x = torch.randn(1, 1, 32)
        ref, fast = self._both(layer, x)
        self.assertEqual(fast.shape, (1, 1, 48))
        self.assertTrue(torch.allclose(ref, fast, atol=1e-6, rtol=1e-5))

    def test_matches_loop_naive_shared_gate(self):
        """naive gate: shared weights are all-ones, still folded into the fast path."""
        torch.manual_seed(5)
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=6, top_k=2, rank=4, alpha=8,
                              dropout=0.0, num_shared_experts=2, shared_rank=8,
                              shared_expert_gate_mode="naive")
        self._randomize(layer)
        x = torch.randn(2, 4, 32)
        ref, fast = self._both(layer, x)
        self.assertTrue(torch.allclose(ref, fast, atol=1e-6, rtol=1e-5),
                        f"max abs diff {(ref - fast).abs().max().item():.3e}")

    def test_matches_loop_hydralora_shape(self):
        """HydraLoRA baseline: shared A, N B heads, dense router (top_k == N)."""
        torch.manual_seed(6)
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=3, top_k=3, rank=8, alpha=16,
                              dropout=0.0, use_shared_expert=False, share_routing_A=True)
        self._randomize(layer)
        # Tied A: randomizing through the list must not have broken the tie.
        self.assertIs(layer.routing_experts[0].lora_A, layer.routing_experts[2].lora_A)
        x = torch.randn(2, 5, 32)
        ref, fast = self._both(layer, x)
        self.assertTrue(torch.allclose(ref, fast, atol=1e-6, rtol=1e-5))

    def test_tied_A_survives_save_load_roundtrip(self):
        """HydraLoRA's tied A is saved once; loading must restore all three heads."""
        import tempfile
        from train_moe_apr import save_moe_state, load_moe_state

        torch.manual_seed(7)
        # Both layers must wrap the SAME frozen base: save_moe_state stores only
        # the adapter, so a freshly initialised base would differ on its own.
        base = nn.Linear(32, 48, bias=False)

        def build():
            return MoELoRALinear(base, num_routing_experts=3, top_k=3, rank=8, alpha=16,
                                 dropout=0.0, use_shared_expert=False, share_routing_A=True)

        src = build()
        self._randomize(src)
        src.eval()
        x = torch.randn(2, 4, 32)
        with torch.no_grad():
            expected = src(x)

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            save_moe_state(src, tmp.name)
            dst = build()
            load_moe_state(dst, tmp.name, strict=True)  # strict: no unexplained gaps
        dst.eval()
        with torch.no_grad():
            got = dst(x)
        self.assertTrue(torch.allclose(expected, got, atol=1e-6),
                        "tied lora_A did not round-trip through save/load")

    def test_training_mode_uses_loop(self):
        """Dropout must stay per-expert during training -> never take the fast path."""
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=4, top_k=2, rank=4, alpha=8, dropout=0.1)
        layer.train()
        self.assertFalse(layer._fast_path_available())
        layer.eval()
        self.assertTrue(layer._fast_path_available())

    def test_train_switch_invalidates_cache(self):
        """Weights change during training, so the concatenated cache must be dropped."""
        torch.manual_seed(4)
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, num_routing_experts=4, top_k=2, rank=4, alpha=8, dropout=0.0)
        self._randomize(layer)
        layer.eval()
        with torch.no_grad():
            layer(torch.randn(1, 3, 32))
        self.assertIsNotNone(layer._fast_cat)
        layer.train()
        self.assertIsNone(layer._fast_cat)

        # After a weight update the rebuilt cache must reflect the new weights.
        with torch.no_grad():
            layer.routing_experts[0].lora_B.weight.add_(1.0)
        x = torch.randn(1, 3, 32)
        ref, fast = self._both(layer, x)
        self.assertTrue(torch.allclose(ref, fast, atol=1e-6, rtol=1e-5))


class TestBranchAblation(unittest.TestCase):
    """Inference-time branch ablation (``MOE_ABLATE`` / ``MOE_ABLATE_NORM``).

    These back the RQ4 "which branch carries the cross-lingual prior" analysis:
    the shared branch and the routing branch are zeroed in turn, in both the
    fast concatenated path and the per-expert loop.
    """

    S3_KW = dict(num_routing_experts=11, top_k=1, rank=4, alpha=8, dropout=0.0,
                 num_shared_experts=3, shared_rank=4)

    def _layer(self, seed=0, **kw):
        torch.manual_seed(seed)
        kwargs = dict(self.S3_KW)
        kwargs.update(kw)
        base = nn.Linear(32, 48, bias=False)
        layer = MoELoRALinear(base, **kwargs)
        for e in list(layer.routing_experts) + list(layer.shared_experts):
            torch.nn.init.normal_(e.lora_A.weight, std=0.05)
            torch.nn.init.normal_(e.lora_B.weight, std=0.05)
        layer.eval()
        return layer

    @staticmethod
    def _fwd(layer, x, fast=True):
        prev = os.environ.get("MOE_FAST_INFER")
        os.environ["MOE_FAST_INFER"] = "1" if fast else "0"
        try:
            with torch.no_grad():
                return layer(x).clone()
        finally:
            if prev is None:
                os.environ.pop("MOE_FAST_INFER", None)
            else:
                os.environ["MOE_FAST_INFER"] = prev

    def setUp(self):
        self._saved = get_moe_ablation()

    def tearDown(self):
        set_moe_ablation(*self._saved)

    # -- defaults ----------------------------------------------------------- #

    def test_default_is_none_and_changes_nothing(self):
        """With no env set the layer must be bit-identical to the pre-ablation code."""
        layer = self._layer()
        x = torch.randn(2, 6, 32)
        set_moe_ablation("none", "drop_renorm")
        y_default = self._fwd(layer, x)
        # Explicitly bypass the hook to get the untouched reference.
        with torch.no_grad():
            base_out = layer.base(x)
            rl, w, idx, sw = layer._compute_routing_weights(x)
        self.assertIsNotNone(sw)
        total = w.sum(-1) + sw.sum(-1)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))
        self.assertFalse(torch.allclose(y_default, base_out))  # adapter is live

        # Ablation "none" under both norms is the same tensor.
        set_moe_ablation("none", "drop")
        self.assertTrue(torch.equal(y_default, self._fwd(layer, x)))

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            set_moe_ablation("banana")
        with self.assertRaises(ValueError):
            set_moe_ablation("shared", "renormalise")

    def test_context_manager_restores(self):
        set_moe_ablation("none", "drop_renorm")
        with moe_ablation("routing", "drop"):
            self.assertEqual(get_moe_ablation(), ("routing", "drop"))
        self.assertEqual(get_moe_ablation(), ("none", "drop_renorm"))

    # -- training path is untouched ----------------------------------------- #

    def test_training_mode_ignores_ablation(self):
        layer = self._layer()
        x = torch.randn(2, 6, 32)
        layer.train()
        torch.manual_seed(123)
        with moe_ablation("none"):
            with torch.no_grad():
                y_full = layer(x).clone()
        for mode in ("shared", "routing"):
            for norm in ("drop", "drop_renorm"):
                with moe_ablation(mode, norm):
                    with torch.no_grad():
                        y = layer(x)
                self.assertTrue(torch.equal(y_full, y),
                                f"training forward changed under MOE_ABLATE={mode}/{norm}")
        layer.eval()

    # -- the branches really do drop out ------------------------------------ #

    def test_no_shared_removes_shared_contribution(self):
        """no_shared must equal a hand-built forward that only sums routing experts."""
        layer = self._layer(seed=1)
        x = torch.randn(2, 5, 32)
        with moe_ablation("shared", "drop"):
            y = self._fwd(layer, x)
            self.assertTrue(torch.allclose(
                layer._cached_shared_weight, torch.zeros_like(layer._cached_shared_weight)))
        with torch.no_grad():
            _, w, idx, _ = layer._compute_routing_weights(x)  # ablated weights (shared==0)
            manual = layer.base(x)
            one_hot = F.one_hot(idx, num_classes=layer.num_routing_experts).to(w.dtype)
            wpe = (w.unsqueeze(-1) * one_hot).sum(dim=-2)
            for i, e in enumerate(layer.routing_experts):
                manual = manual + wpe[..., i : i + 1] * e(x)
        self.assertTrue(torch.allclose(y, manual, atol=1e-6),
                        f"max abs diff {(y - manual).abs().max().item():.3e}")

        # Perturbing the shared experts must now be a no-op.
        with torch.no_grad():
            for e in layer.shared_experts:
                e.lora_B.weight.add_(5.0)
        layer._fast_cat = None
        with moe_ablation("shared", "drop"):
            self.assertTrue(torch.allclose(y, self._fwd(layer, x), atol=1e-6))

    def test_no_routing_removes_routing_contribution(self):
        layer = self._layer(seed=2)
        x = torch.randn(2, 5, 32)
        with moe_ablation("routing", "drop"):
            y = self._fwd(layer, x)
            self.assertTrue(torch.allclose(
                layer._cached_top_k_weights, torch.zeros_like(layer._cached_top_k_weights)))
        with torch.no_grad():
            _, _, _, sw = layer._compute_routing_weights(x)
            manual = layer.base(x)
            for s, e in enumerate(layer.shared_experts):
                manual = manual + sw[..., s : s + 1] * e(x)
        self.assertTrue(torch.allclose(y, manual, atol=1e-6),
                        f"max abs diff {(y - manual).abs().max().item():.3e}")

        with torch.no_grad():
            for e in layer.routing_experts:
                e.lora_B.weight.add_(5.0)
        layer._fast_cat = None
        with moe_ablation("routing", "drop"):
            self.assertTrue(torch.allclose(y, self._fwd(layer, x), atol=1e-6))

    def test_both_branches_dropped_recovers_base(self):
        """Sanity: shared+routing are the only adapter terms, nothing else leaks."""
        layer = self._layer(seed=8)
        x = torch.randn(2, 4, 32)
        with torch.no_grad():
            base_out = layer.base(x)
        # Zeroing one branch at a time and summing the deltas must reconstruct
        # the full adapter delta (the branches are additive).
        with moe_ablation("shared", "drop"):
            y_routing_only = self._fwd(layer, x)
        with moe_ablation("routing", "drop"):
            y_shared_only = self._fwd(layer, x)
        with moe_ablation("none"):
            y_full = self._fwd(layer, x)
        recon = base_out + (y_routing_only - base_out) + (y_shared_only - base_out)
        self.assertTrue(torch.allclose(y_full, recon, atol=1e-5),
                        f"max abs diff {(y_full - recon).abs().max().item():.3e}")

    # -- renormalization ----------------------------------------------------- #

    def test_renorm_makes_surviving_weights_sum_to_one(self):
        layer = self._layer(seed=3)
        x = torch.randn(2, 5, 32)
        with moe_ablation("shared", "drop_renorm"):
            self._fwd(layer, x)
            tot = layer._cached_top_k_weights.sum(-1)
            self.assertTrue(torch.allclose(tot, torch.ones_like(tot), atol=1e-5))
            self.assertTrue(torch.allclose(
                layer._cached_shared_weight, torch.zeros_like(layer._cached_shared_weight)))
        with moe_ablation("routing", "drop_renorm"):
            self._fwd(layer, x)
            tot = layer._cached_shared_weight.sum(-1)
            self.assertTrue(torch.allclose(tot, torch.ones_like(tot), atol=1e-5))
            self.assertTrue(torch.allclose(
                layer._cached_top_k_weights, torch.zeros_like(layer._cached_top_k_weights)))

    def test_drop_vs_renorm_differ_and_scale_correctly(self):
        """renorm output delta = drop output delta / (surviving weight mass)."""
        layer = self._layer(seed=4, num_shared_experts=1, top_k=1)
        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            base_out = layer.base(x)
        with moe_ablation("routing", "drop"):
            y_drop = self._fwd(layer, x)
            share = layer._cached_shared_weight.sum(-1, keepdim=True).clone()
        with moe_ablation("routing", "drop_renorm"):
            y_renorm = self._fwd(layer, x)
        self.assertFalse(torch.allclose(y_drop, y_renorm))
        # With one shared expert, renorm sets its weight to exactly 1.
        self.assertTrue(torch.allclose((y_renorm - base_out) * share, y_drop - base_out, atol=1e-6))

    def test_naive_gate_renorm_is_noop_on_shared(self):
        """Naive shared gates are constants, not a budget -> never rescaled."""
        layer = self._layer(seed=9, shared_expert_gate_mode="naive")
        x = torch.randn(2, 4, 32)
        with moe_ablation("routing", "drop_renorm"):
            y_renorm = self._fwd(layer, x)
            self.assertTrue(torch.allclose(
                layer._cached_shared_weight, torch.ones_like(layer._cached_shared_weight)))
        with moe_ablation("routing", "drop"):
            y_drop = self._fwd(layer, x)
        self.assertTrue(torch.equal(y_renorm, y_drop))

    def test_ablate_shared_on_layer_without_shared_is_noop(self):
        layer = self._layer(seed=5, use_shared_expert=False)
        x = torch.randn(2, 4, 32)
        with moe_ablation("none"):
            y_full = self._fwd(layer, x)
        with moe_ablation("shared", "drop_renorm"):
            self.assertTrue(torch.equal(y_full, self._fwd(layer, x)))

    # -- the two forward paths agree under ablation -------------------------- #

    def test_fast_path_matches_loop_under_every_condition(self):
        for seed, kw in ((6, {}), (7, dict(num_routing_experts=6, top_k=2, num_shared_experts=2))):
            layer = self._layer(seed=seed, **kw)
            x = torch.randn(2, 5, 32)
            for mode in ("none", "shared", "routing"):
                for norm in ("drop", "drop_renorm"):
                    with moe_ablation(mode, norm):
                        ref = self._fwd(layer, x, fast=False)
                        fast = self._fwd(layer, x, fast=True)
                    self.assertTrue(
                        torch.allclose(ref, fast, atol=1e-6, rtol=1e-5),
                        f"seed={seed} {mode}/{norm}: max abs diff "
                        f"{(ref - fast).abs().max().item():.3e}",
                    )

    def test_single_token_decode_under_ablation(self):
        """Autoregressive decode step (batch=1, seq=1) -- the real inference shape."""
        layer = self._layer(seed=10)
        x = torch.randn(1, 1, 32)
        for mode in ("shared", "routing"):
            with moe_ablation(mode, "drop_renorm"):
                ref = self._fwd(layer, x, fast=False)
                fast = self._fwd(layer, x, fast=True)
            self.assertEqual(fast.shape, (1, 1, 48))
            self.assertTrue(torch.allclose(ref, fast, atol=1e-6, rtol=1e-5))


class TestAblationEnvParsing(unittest.TestCase):
    def setUp(self):
        self._saved = get_moe_ablation()
        self._env = {k: os.environ.get(k) for k in ("MOE_ABLATE", "MOE_ABLATE_NORM")}

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        set_moe_ablation(*self._saved)

    def test_env_roundtrip(self):
        from moe_apr.moe_layer import refresh_moe_ablation_from_env
        os.environ.pop("MOE_ABLATE", None)
        os.environ.pop("MOE_ABLATE_NORM", None)
        self.assertEqual(refresh_moe_ablation_from_env(), ("none", "drop_renorm"))
        os.environ["MOE_ABLATE"] = "Shared"      # case/space tolerant
        os.environ["MOE_ABLATE_NORM"] = " drop "
        self.assertEqual(refresh_moe_ablation_from_env(), ("shared", "drop"))
        os.environ["MOE_ABLATE"] = "nonsense"
        with self.assertRaises(ValueError):
            refresh_moe_ablation_from_env()


if __name__ == "__main__":
    unittest.main()
