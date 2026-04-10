from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConstantGate:
    scale: float = 1.0

    def __call__(
        self,
        latents: torch.Tensor,
        step: int,
        estimator: object,
        good_clean_samples: torch.Tensor,
        bad_clean_samples: torch.Tensor,
    ) -> torch.Tensor:
        return latents.new_full((latents.shape[0],), float(self.scale))


@dataclass(frozen=True)
class DensityRatioGate:
    """
    Down-weight steering when a sample already looks more good-like than bad-like.
    """

    temperature: float = 1.0
    bias: float = 0.0
    min_scale: float = 0.0
    max_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.max_scale < self.min_scale:
            raise ValueError("max_scale must be >= min_scale")

    def __call__(
        self,
        latents: torch.Tensor,
        step: int,
        estimator: object,
        good_clean_samples: torch.Tensor,
        bad_clean_samples: torch.Tensor,
    ) -> torch.Tensor:
        if bad_clean_samples.shape[0] == 0:
            return latents.new_ones((latents.shape[0],))

        log_good = estimator.mixture_log_density(latents, step, good_clean_samples)
        log_bad = estimator.mixture_log_density(latents, step, bad_clean_samples)
        logits = (log_bad - log_good + self.bias) / self.temperature
        gate = torch.sigmoid(logits)
        if self.min_scale == 0.0 and self.max_scale == 1.0:
            return gate
        return self.min_scale + (self.max_scale - self.min_scale) * gate
