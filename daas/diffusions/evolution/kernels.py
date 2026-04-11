from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol

import torch


def _compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


class Kernel(Protocol):
    def kernel_matrix(self, particles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def score_correction(self, particles: torch.Tensor, kernel_matrix: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class RBFKernel:
    bandwidth: Optional[float] = None
    min_bandwidth: float = 1e-6

    def __post_init__(self) -> None:
        if self.bandwidth is not None and self.bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")
        if self.min_bandwidth <= 0:
            raise ValueError("min_bandwidth must be > 0")

    def resolve_bandwidth(self, particles: torch.Tensor) -> torch.Tensor:
        if self.bandwidth is not None:
            return particles.new_tensor(float(self.bandwidth))

        num_particles = particles.shape[0]
        if num_particles < 2:
            return particles.new_tensor(1.0)

        squared_distances = torch.cdist(particles, particles).square()
        upper = squared_distances[torch.triu_indices(num_particles, num_particles, offset=1).unbind(0)]
        if upper.numel() == 0:
            return particles.new_tensor(1.0)

        median_distance = torch.median(upper)
        heuristic = median_distance / math.log(num_particles + 1.0)
        return heuristic.clamp_min(self.min_bandwidth)

    def kernel_matrix(self, particles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bandwidth = self.resolve_bandwidth(particles)
        squared_distances = torch.cdist(particles, particles).square()
        return torch.exp(-squared_distances / bandwidth), bandwidth

    def score_correction(
        self,
        particles: torch.Tensor,
        kernel_matrix: torch.Tensor,
        bandwidth: torch.Tensor,
    ) -> torch.Tensor:
        pairwise_diff = particles[:, None, :] - particles[None, :, :]
        return (2.0 / bandwidth) * (pairwise_diff * kernel_matrix.unsqueeze(-1)).sum(dim=1)
