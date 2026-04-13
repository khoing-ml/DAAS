from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


StepLike = Union[int, torch.Tensor]


@dataclass(frozen=True)
class DiffusionSchedule:
    """Forward-process coefficients used to approximate p(x_t | x_0)."""

    betas: torch.Tensor
    kind: str = "custom"

    def __post_init__(self) -> None:
        betas = torch.as_tensor(self.betas, dtype=torch.float32)
        if betas.ndim != 1:
            raise ValueError("betas must be a 1D tensor")
        if betas.numel() == 0:
            raise ValueError("betas must contain at least one timestep")
        if torch.any((betas < 0) | (betas >= 1)):
            raise ValueError("betas must satisfy 0 <= beta < 1")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "alphas", alphas)
        object.__setattr__(self, "alpha_bars", alpha_bars)

    @property
    def num_steps(self) -> int:
        return int(self.betas.shape[0])

    @classmethod
    def from_betas(cls, betas: torch.Tensor, kind: str = "custom") -> "DiffusionSchedule":
        return cls(betas=betas, kind=kind)

    @classmethod
    def linear(
        cls,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        num_train_timesteps: int = 1000,
    ) -> "DiffusionSchedule":
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        return cls.from_betas(betas, kind="linear")

    @classmethod
    def scaled_linear(
        cls,
        beta_start: float = 8.5e-4,
        beta_end: float = 1.2e-2,
        num_train_timesteps: int = 1000,
    ) -> "DiffusionSchedule":
        sqrt_betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32)
        betas = sqrt_betas.square()
        return cls.from_betas(betas, kind="scaled_linear")

    @classmethod
    def from_diffusers_scheduler(cls, scheduler: object) -> "DiffusionSchedule":
        if not hasattr(scheduler, "betas"):
            raise ValueError("scheduler must expose a `betas` attribute")
        config = getattr(scheduler, "config", None)
        kind = getattr(config, "beta_schedule", "diffusers") if config is not None else "diffusers"
        return cls.from_betas(torch.as_tensor(scheduler.betas), kind=str(kind))

    def resolve_step(self, step: StepLike) -> int:
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError("step tensor must be scalar")
            step = int(step.item())
        step = int(step)
        if step < 0 or step >= self.num_steps:
            raise IndexError(f"step {step} is out of range for {self.num_steps} timesteps")
        return step

    def alpha_bar(self, step: StepLike, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        value = self.alpha_bars[self.resolve_step(step)]
        if like is None:
            return value
        return value.to(device=like.device, dtype=like.dtype)

    def signal_scale(self, step: StepLike, like: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar(step, like=like).sqrt()

    def noise_variance(self, step: StepLike, like: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (1.0 - self.alpha_bar(step, like=like)).clamp_min(eps)

    def noise_std(self, step: StepLike, like: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return self.noise_variance(step, like=like, eps=eps).sqrt()

    def mean_from_clean(self, clean_samples: torch.Tensor, step: StepLike) -> torch.Tensor:
        return self.signal_scale(step, like=clean_samples) * clean_samples

    def score_from_clean(self, latents: torch.Tensor, clean_samples: torch.Tensor, step: StepLike) -> torch.Tensor:
        if latents.shape != clean_samples.shape:
            raise ValueError("latents and clean_samples must share the same shape")
        mean = self.mean_from_clean(clean_samples, step)
        variance = self.noise_variance(step, like=latents)
        return -(latents - mean) / variance
