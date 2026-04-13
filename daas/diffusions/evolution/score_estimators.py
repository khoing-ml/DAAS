from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol

import torch

from daas.diffusions.evolution.kernels import _compute_dtype
from daas.diffusions.evolution.schedules import DiffusionSchedule


class ScoreEstimator(Protocol):
    def mixture_log_density(self, latents: torch.Tensor, step: int, clean_refs: torch.Tensor) -> torch.Tensor: ...

    def score(self, latents: torch.Tensor, step: int, clean_refs: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class GoodSetScoreEstimator:
    """Approximate ∇ log q_t(x) from a set of good clean samples."""

    schedule: DiffusionSchedule
    temperature: float = 1.0
    chunk_size: Optional[int] = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def _prepare(
        self,
        latents: torch.Tensor,
        clean_refs: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if clean_refs.ndim != latents.ndim:
            raise ValueError("latents and clean_refs must have the same rank")
        if tuple(clean_refs.shape[1:]) != tuple(latents.shape[1:]):
            raise ValueError("latents and clean_refs must share the same per-sample shape")
        if clean_refs.shape[0] == 0:
            raise ValueError("clean_refs must contain at least one reference sample")

        compute_dtype = _compute_dtype(latents)
        flat_queries = latents.flatten(start_dim=1).to(dtype=compute_dtype)
        flat_refs = clean_refs.to(device=latents.device, dtype=compute_dtype).flatten(start_dim=1)
        signal_scale = self.schedule.signal_scale(step, like=flat_queries).to(dtype=compute_dtype)
        means = signal_scale * flat_refs
        variance = self.schedule.noise_variance(step, like=flat_queries, eps=self.eps).to(dtype=compute_dtype)
        return flat_queries, means, variance

    def mixture_log_density(self, latents: torch.Tensor, step: int, clean_refs: torch.Tensor) -> torch.Tensor:
        flat_queries, means, variance = self._prepare(latents, clean_refs, step)
        chunk_size = self.chunk_size or flat_queries.shape[0]
        outputs = []

        for query_chunk in flat_queries.split(chunk_size):
            squared_distances = torch.cdist(query_chunk, means).square()
            logits = -squared_distances / (2.0 * variance * self.temperature)
            outputs.append(torch.logsumexp(logits, dim=1) - math.log(means.shape[0]))

        return torch.cat(outputs, dim=0).to(dtype=latents.dtype)

    def score(self, latents: torch.Tensor, step: int, clean_refs: torch.Tensor) -> torch.Tensor:
        flat_queries, means, variance = self._prepare(latents, clean_refs, step)
        chunk_size = self.chunk_size or flat_queries.shape[0]
        outputs = []

        for query_chunk in flat_queries.split(chunk_size):
            squared_distances = torch.cdist(query_chunk, means).square()
            logits = -squared_distances / (2.0 * variance * self.temperature)
            weights = torch.softmax(logits, dim=1)
            weighted_means = weights @ means
            outputs.append(-(query_chunk - weighted_means) / variance)

        flat_scores = torch.cat(outputs, dim=0).to(dtype=latents.dtype)
        return flat_scores.reshape_as(latents)


@dataclass(frozen=True)
class KernelDensityScoreEstimator:
    """
    Estimate the good-set score by fitting a Gaussian KDE in clean space and
    propagating it through the diffusion schedule.
    """

    schedule: DiffusionSchedule
    bandwidth: Optional[float] = None
    min_bandwidth: float = 1e-6
    chunk_size: Optional[int] = None
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.bandwidth is not None and self.bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")
        if self.min_bandwidth <= 0:
            raise ValueError("min_bandwidth must be > 0")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def _resolve_bandwidth(self, flat_refs: torch.Tensor) -> torch.Tensor:
        if self.bandwidth is not None:
            return flat_refs.new_tensor(float(self.bandwidth))

        num_refs = flat_refs.shape[0]
        if num_refs < 2:
            return flat_refs.new_tensor(1.0)

        squared_distances = torch.cdist(flat_refs, flat_refs).square()
        upper = squared_distances[torch.triu_indices(num_refs, num_refs, offset=1).unbind(0)]
        if upper.numel() == 0:
            return flat_refs.new_tensor(1.0)

        median_distance = torch.median(upper)
        heuristic = median_distance / math.log(num_refs + 1.0)
        return heuristic.clamp_min(self.min_bandwidth).sqrt()

    def _prepare(
        self,
        latents: torch.Tensor,
        clean_refs: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if clean_refs.ndim != latents.ndim:
            raise ValueError("latents and clean_refs must have the same rank")
        if tuple(clean_refs.shape[1:]) != tuple(latents.shape[1:]):
            raise ValueError("latents and clean_refs must share the same per-sample shape")
        if clean_refs.shape[0] == 0:
            raise ValueError("clean_refs must contain at least one reference sample")

        compute_dtype = _compute_dtype(latents)
        flat_queries = latents.flatten(start_dim=1).to(dtype=compute_dtype)
        flat_refs = clean_refs.to(device=latents.device, dtype=compute_dtype).flatten(start_dim=1)
        signal_scale = self.schedule.signal_scale(step, like=flat_queries).to(dtype=compute_dtype)
        diffusion_variance = self.schedule.noise_variance(step, like=flat_queries, eps=self.eps).to(dtype=compute_dtype)
        bandwidth = self._resolve_bandwidth(flat_refs).to(dtype=compute_dtype)
        total_variance = diffusion_variance + signal_scale.square() * bandwidth.square()
        means = signal_scale * flat_refs
        return flat_queries, means, total_variance

    def mixture_log_density(self, latents: torch.Tensor, step: int, clean_refs: torch.Tensor) -> torch.Tensor:
        flat_queries, means, total_variance = self._prepare(latents, clean_refs, step)
        chunk_size = self.chunk_size or flat_queries.shape[0]
        dim = means.shape[1]
        normalization = -0.5 * dim * torch.log(2.0 * torch.pi * total_variance)
        outputs = []

        for query_chunk in flat_queries.split(chunk_size):
            squared_distances = torch.cdist(query_chunk, means).square()
            logits = -squared_distances / (2.0 * total_variance)
            outputs.append(torch.logsumexp(logits, dim=1) - math.log(means.shape[0]) + normalization)

        return torch.cat(outputs, dim=0).to(dtype=latents.dtype)

    def score(self, latents: torch.Tensor, step: int, clean_refs: torch.Tensor) -> torch.Tensor:
        flat_queries, means, total_variance = self._prepare(latents, clean_refs, step)
        chunk_size = self.chunk_size or flat_queries.shape[0]
        outputs = []

        for query_chunk in flat_queries.split(chunk_size):
            squared_distances = torch.cdist(query_chunk, means).square()
            logits = -squared_distances / (2.0 * total_variance)
            weights = torch.softmax(logits, dim=1)
            weighted_means = weights @ means
            outputs.append(-(query_chunk - weighted_means) / total_variance)

        flat_scores = torch.cat(outputs, dim=0).to(dtype=latents.dtype)
        return flat_scores.reshape_as(latents)
