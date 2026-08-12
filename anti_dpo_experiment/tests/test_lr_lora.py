import sys
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lr_lora import LearnableRankLoRALinear


@unittest.skipIf(torch is None, "torch is not installed")
class TestLearnableRankLoRA(unittest.TestCase):
    def test_zero_initialized_adapter_matches_frozen_base(self):
        base = nn.Linear(4, 3)
        adapter = LearnableRankLoRALinear(base, rank=2)
        inputs = torch.randn(2, 4)
        self.assertTrue(torch.allclose(adapter(inputs), base(inputs)))

    def test_effective_update_and_stable_rank_are_finite(self):
        adapter = LearnableRankLoRALinear(nn.Linear(4, 3), rank=2)
        self.assertTrue(torch.isfinite(adapter.effective_update).all())
        self.assertTrue(torch.isfinite(adapter.stable_rank()))
