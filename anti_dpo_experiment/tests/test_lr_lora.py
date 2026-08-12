import sys
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lr_lora import LearnableRankLoRALinear, load_lr_lora_adapter, save_lr_lora_adapter


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

    def test_adapter_follows_half_precision_base_dtype(self):
        base = nn.Linear(4, 3).half()
        adapter = LearnableRankLoRALinear(base, rank=2)
        inputs = torch.randn(2, 4).half()
        outputs = adapter(inputs)
        self.assertEqual(outputs.dtype, torch.float16)
        self.assertEqual(adapter.effective_update.dtype, torch.float32)

    def test_zero_initialization_starts_by_learning_sinc_amplitudes(self):
        adapter = LearnableRankLoRALinear(nn.Linear(4, 3), rank=2)
        adapter(torch.randn(2, 4)).square().mean().backward()
        self.assertGreater(adapter.nonlinearity.amplitudes.grad.abs().sum().item(), 0.0)
        self.assertEqual(adapter.lora_a.grad.abs().sum().item(), 0.0)
        self.assertEqual(adapter.lora_b.grad.abs().sum().item(), 0.0)

    def test_adapter_checkpoint_contains_reconstruction_metadata(self):
        module = nn.Module()
        module.projection = LearnableRankLoRALinear(nn.Linear(4, 3), rank=2, num_basis=4)
        with tempfile.TemporaryDirectory() as directory:
            save_lr_lora_adapter(module, directory, "test/model", 2, 4.0, 4, ["projection"])
            state = torch.load(Path(directory) / "lr_lora_adapter.pt", weights_only=True)
            config = json.loads((Path(directory) / "lr_lora_adapter_config.json").read_text())
        self.assertIn("projection.lora_a", state)
        self.assertEqual(config["target_modules"], ["projection"])

    def test_adapter_checkpoint_round_trip(self):
        source = nn.Module()
        source.projection = LearnableRankLoRALinear(nn.Linear(4, 3), rank=2, num_basis=4)
        with torch.no_grad():
            source.projection.nonlinearity.amplitudes.uniform_(-0.2, 0.2)
        target = nn.Module()
        target.projection = LearnableRankLoRALinear(nn.Linear(4, 3), rank=2, num_basis=4)
        with tempfile.TemporaryDirectory() as directory:
            save_lr_lora_adapter(source, directory, "test/model", 2, 4.0, 4, ["projection"])
            load_lr_lora_adapter(target, directory)
        self.assertTrue(torch.equal(source.projection.lora_a, target.projection.lora_a))
        self.assertTrue(torch.equal(source.projection.nonlinearity.amplitudes, target.projection.nonlinearity.amplitudes))
