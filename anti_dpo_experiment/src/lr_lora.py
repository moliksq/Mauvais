"""Experimental Learnable-Rank LoRA from the supplied LR-LoRA paper."""

from __future__ import annotations

import math

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
        def __init__(self, num_basis: int = 8, interval: float = 2.0, omega0: float = 1.0) -> None:
            super().__init__()
            if num_basis < 1 or interval <= 0 or omega0 <= 0:
                raise ValueError("num_basis, interval, and omega0 must be positive")
            self.register_buffer("centers", torch.linspace(-interval, interval, num_basis))
            self.amplitudes = nn.Parameter(torch.zeros(num_basis))
            self.raw_bandwidths = nn.Parameter(torch.full((num_basis,), math.log(math.expm1(omega0))))

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
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
            self.scaling = alpha / rank
            self.nonlinearity = LearnableSinc(num_basis=num_basis)
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            # LR-LoRA applies phi to the materialized weight update, as in the
            # supplied paper: Delta W = phi(BA), rather than phi(xBA).
            update = self.effective_update
            return self.base(value) + F.linear(value, update)

        @property
        def effective_update(self) -> torch.Tensor:
            return self.scaling * self.nonlinearity(self.lora_b @ self.lora_a)

        def stable_rank(self) -> torch.Tensor:
            update = self.effective_update.float()
            singular_values = torch.linalg.svdvals(update)
            return update.square().sum() / singular_values.square().max().clamp_min(1e-12)
else:
    class LearnableRankLoRALinear:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch is required for LR-LoRA") from _IMPORT_ERROR
