import unittest

import torch
import torch.nn as nn

from moe_apr.moe_layer import MoELoRALinear
from moe_apr.moe_metrics import moe_routing_stats


class TestMoEMetrics(unittest.TestCase):
    def test_routing_stats_after_forward(self):
        torch.manual_seed(0)
        layer = MoELoRALinear(
            nn.Linear(16, 32),
            num_routing_experts=4,
            top_k=2,
            rank=4,
            alpha=8,
            dropout=0.0,
            use_shared_expert=True,
            shared_expert_gate_mode="adaptive",
        )
        x = torch.randn(2, 8, 16)
        labels = torch.full((2, 8), -100)
        labels[:, -3:] = 1
        route_ids = torch.tensor([0, 2])
        layer(x)

        stats = moe_routing_stats(
            [layer],
            route_ids=route_ids,
            label_mask=labels,
        )
        self.assertIn("router_entropy", stats)
        self.assertIn("load_imbalance", stats)
        self.assertIn("shared_gate", stats)
        self.assertIn("route_purity", stats)
        self.assertEqual(len([k for k in stats if k.startswith("expert_util_")]), 4)
        self.assertGreater(stats["router_entropy"], 0.0)

    def test_empty_when_no_cache(self):
        layer = MoELoRALinear(nn.Linear(8, 8), num_routing_experts=2, top_k=1, rank=2, alpha=4)
        self.assertEqual(moe_routing_stats([layer]), {})


if __name__ == "__main__":
    unittest.main()
