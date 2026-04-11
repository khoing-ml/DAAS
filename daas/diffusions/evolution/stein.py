from __future__ import annotations

from dataclasses import dataclass

import torch

from daas.diffusions.evolution.kernels import Kernel, _compute_dtype


@dataclass(frozen=True)
class SteinVectorField:
    kernel: Kernel

    def vector_field(self, particles: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
        if particles.shape != target_score.shape:
            raise ValueError("particles and target_score must share the same shape")

        compute_dtype = _compute_dtype(particles)
        flat_particles = particles.flatten(start_dim=1).to(dtype=compute_dtype)
        flat_score = target_score.flatten(start_dim=1).to(dtype=compute_dtype)

        num_particles = flat_particles.shape[0]
        if num_particles == 1:
            return target_score

        kernel_matrix, bandwidth = self.kernel.kernel_matrix(flat_particles)
        attractive = kernel_matrix @ flat_score
        repulsive = self.kernel.score_correction(flat_particles, kernel_matrix, bandwidth)
        field = (attractive + repulsive) / num_particles
        return field.to(dtype=particles.dtype).reshape_as(particles)
