from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def _compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


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

    def vector_field(self, particles: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
        if particles.shape != target_score.shape:
            raise ValueError("particles and target_score must share the same shape")

        compute_dtype = _compute_dtype(particles)
        flat_particles = particles.flatten(start_dim=1).to(dtype=compute_dtype)
        flat_score = target_score.flatten(start_dim=1).to(dtype=compute_dtype)

        num_particles = flat_particles.shape[0]
        if num_particles == 1:
            return target_score

        bandwidth = self.resolve_bandwidth(flat_particles)
        squared_distances = torch.cdist(flat_particles, flat_particles).square()
        kernel_matrix = torch.exp(-squared_distances / bandwidth)
        pairwise_diff = flat_particles[:, None, :] - flat_particles[None, :, :]

        attractive = kernel_matrix @ flat_score
        repulsive = (2.0 / bandwidth) * (pairwise_diff * kernel_matrix.unsqueeze(-1)).sum(dim=1)
        field = (attractive + repulsive) / num_particles
        return field.to(dtype=particles.dtype).reshape_as(particles)
