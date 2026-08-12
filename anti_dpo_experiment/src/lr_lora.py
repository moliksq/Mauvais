"""Experimental Learnable-Rank LoRA from the supplied LR-LoRA paper."""

from __future__ import annotations

import math
import json
from pathlib import Path

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    nn = object  # type: ignore[assignment]
    F = None
    _IMPORT_ERROR = exc


if torch is not None:
    class LearnableSinc(nn.Module):
        def __init__(
            self,
            num_basis: int = 8,
            interval: float = 2.0,
            omega0: float = 1.0,
            device=None,
            dtype=None,
        ) -> None:
            super().__init__()
            if num_basis < 1 or interval <= 0 or omega0 <= 0:
                raise ValueError("num_basis, interval, and omega0 must be positive")
            # Keep learned basis parameters in FP32 even when the frozen model is
            # FP16. Their updates are small and otherwise underflow on a T4.
            factory_kwargs = {"device": device, "dtype": torch.float32}
            self.register_buffer("centers", torch.linspace(-interval, interval, num_basis, **factory_kwargs))
            self.amplitudes = nn.Parameter(torch.zeros(num_basis, **factory_kwargs))
            self.raw_bandwidths = nn.Parameter(torch.full((num_basis,), math.log(math.expm1(omega0)), **factory_kwargs))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            bandwidths = F.softplus(self.raw_bandwidths)
            argument = bandwidths.view(*([1] * value.ndim), -1) * (
                value.unsqueeze(-1) - self.centers.view(*([1] * value.ndim), -1)
            )
            return (torch.sinc(argument / math.pi) * self.amplitudes.view(*([1] * value.ndim), -1)).sum(dim=-1)


    class LearnableRankLoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, num_basis: int = 8) -> None:
            super().__init__()
            if rank < 1:
                raise ValueError("rank must be >= 1")
            self.base = base
            for parameter in base.parameters():
                parameter.requires_grad_(False)
            factory_kwargs = {"device": base.weight.device, "dtype": torch.float32}
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **factory_kwargs))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **factory_kwargs))
            self.scaling = alpha / rank
            self.nonlinearity = LearnableSinc(num_basis=num_basis, device=base.weight.device)
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            # LR-LoRA applies phi to the materialized weight update, as in the
            # supplied paper: Delta W = phi(BA), rather than phi(xBA).
            update = self.effective_update
            adapter_output = F.linear(value.to(dtype=update.dtype), update).to(dtype=value.dtype)
            return self.base(value) + adapter_output

        @property
        def effective_update(self) -> torch.Tensor:
            return self.scaling * self.nonlinearity(self.lora_b @ self.lora_a)

        def stable_rank(self) -> torch.Tensor:
            update = self.effective_update.float()
            singular_values = torch.linalg.svdvals(update)
            return update.square().sum() / singular_values.square().max().clamp_min(1e-12)


    def lr_lora_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        """Return LR-LoRA tensors and buffers, detached on CPU for portability."""

        return {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if ".lora_a" in name or ".lora_b" in name or ".nonlinearity." in name
        }


    def save_lr_lora_adapter(
        model: nn.Module,
        output_dir: str | Path,
        model_name: str,
        rank: int,
        alpha: float,
        num_basis: int,
        target_modules: list[str],
    ) -> None:
        """Save a portable adapter state and reconstruction metadata."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(lr_lora_adapter_state_dict(model), output / "lr_lora_adapter.pt")
        (output / "lr_lora_adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name": model_name,
                    "rank": rank,
                    "alpha": alpha,
                    "num_basis": num_basis,
                    "target_modules": target_modules,
                    "format": "anti_dpo_experiment.lr_lora.v1",
                },
                indent=2,
            ),
            encoding="utf-8",
        )


    def load_lr_lora_adapter(model: nn.Module, checkpoint_dir: str | Path, device=None) -> dict:
        """Load a saved LR-LoRA adapter into an already reconstructed base model.

        Call this after replacing the target ``nn.Linear`` modules with
        ``LearnableRankLoRALinear`` using the saved JSON configuration.
        """

        checkpoint = Path(checkpoint_dir)
        config = json.loads((checkpoint / "lr_lora_adapter_config.json").read_text(encoding="utf-8"))
        state = torch.load(checkpoint / "lr_lora_adapter.pt", map_location=device or "cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        relevant_missing = [name for name in missing if ".lora_" in name or ".nonlinearity." in name]
        if relevant_missing or unexpected:
            raise ValueError(f"LR-LoRA checkpoint does not match reconstructed model: missing={relevant_missing}, unexpected={unexpected}")
        return config
else:
    class LearnableRankLoRALinear:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch is required for LR-LoRA") from _IMPORT_ERROR


    def save_lr_lora_adapter(*args, **kwargs):
        raise ModuleNotFoundError("torch is required for LR-LoRA") from _IMPORT_ERROR


    def load_lr_lora_adapter(*args, **kwargs):
        raise ModuleNotFoundError("torch is required for LR-LoRA") from _IMPORT_ERROR
