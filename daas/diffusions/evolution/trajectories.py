from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch


@dataclass(frozen=True)
class TrajectoryBatch:
    """A set of sampled trajectories and their final clean samples."""

    latents_by_step: Dict[int, torch.Tensor]
    clean_samples: torch.Tensor
    rewards: torch.Tensor
    prompts: Optional[Sequence[str]] = None
    reward_inputs: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.clean_samples.ndim < 2:
            raise ValueError("clean_samples must have batch and feature dimensions")
        if self.rewards.ndim != 1:
            raise ValueError("rewards must be a 1D tensor")
        if not self.latents_by_step:
            raise ValueError("latents_by_step cannot be empty")

        batch_size = int(self.clean_samples.shape[0])
        if self.rewards.shape[0] != batch_size:
            raise ValueError("rewards must match the clean_samples batch size")

        ordered = {int(step): latents for step, latents in sorted(self.latents_by_step.items())}
        object.__setattr__(self, "latents_by_step", ordered)

        clean_shape = tuple(self.clean_samples.shape[1:])
        for step, latents in ordered.items():
            if latents.shape[0] != batch_size:
                raise ValueError(f"step {step} has batch size {latents.shape[0]}, expected {batch_size}")
            if tuple(latents.shape[1:]) != clean_shape:
                raise ValueError(
                    f"step {step} has shape {tuple(latents.shape[1:])}, expected {clean_shape} to match clean_samples"
                )

        if self.prompts is not None and len(self.prompts) != batch_size:
            raise ValueError("prompts must match the clean_samples batch size")
        if self.reward_inputs is not None and self.reward_inputs.shape[0] != batch_size:
            raise ValueError("reward_inputs must match the clean_samples batch size")

    @property
    def num_particles(self) -> int:
        return int(self.clean_samples.shape[0])

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(self.latents_by_step.keys())

    def latents_at(self, step: int) -> torch.Tensor:
        return self.latents_by_step[int(step)]

    def partition(self, threshold_policy: object) -> "PartitionedTrajectories":
        threshold, good_mask, bad_mask = threshold_policy.split(self.rewards)
        return PartitionedTrajectories(
            trajectories=self,
            threshold=threshold,
            good_mask=good_mask,
            bad_mask=bad_mask,
        )


@dataclass(frozen=True)
class PartitionedTrajectories:
    trajectories: TrajectoryBatch
    threshold: torch.Tensor
    good_mask: torch.Tensor
    bad_mask: torch.Tensor

    def __post_init__(self) -> None:
        batch_size = self.trajectories.num_particles
        if self.good_mask.dtype != torch.bool or self.bad_mask.dtype != torch.bool:
            raise ValueError("good_mask and bad_mask must be boolean tensors")
        if self.good_mask.ndim != 1 or self.bad_mask.ndim != 1:
            raise ValueError("good_mask and bad_mask must be 1D tensors")
        if self.good_mask.shape[0] != batch_size or self.bad_mask.shape[0] != batch_size:
            raise ValueError("masks must match the trajectory batch size")
        if torch.any(self.good_mask & self.bad_mask):
            raise ValueError("good_mask and bad_mask must be disjoint")
        if not torch.all(self.good_mask | self.bad_mask):
            raise ValueError("good_mask and bad_mask must cover the full batch")
        if not self.good_mask.any():
            raise ValueError("at least one good trajectory is required")

    @property
    def num_good(self) -> int:
        return int(self.good_mask.sum().item())

    @property
    def num_bad(self) -> int:
        return int(self.bad_mask.sum().item())

    @property
    def good_clean_samples(self) -> torch.Tensor:
        return self.trajectories.clean_samples[self.good_mask]

    @property
    def bad_clean_samples(self) -> torch.Tensor:
        return self.trajectories.clean_samples[self.bad_mask]

    @property
    def good_rewards(self) -> torch.Tensor:
        return self.trajectories.rewards[self.good_mask]

    @property
    def bad_rewards(self) -> torch.Tensor:
        return self.trajectories.rewards[self.bad_mask]

    def good_latents(self, step: int) -> torch.Tensor:
        return self.trajectories.latents_at(step)[self.good_mask]

    def bad_latents(self, step: int) -> torch.Tensor:
        return self.trajectories.latents_at(step)[self.bad_mask]


@dataclass(frozen=True)
class EvolutionState:
    partition: PartitionedTrajectories

    @property
    def threshold(self) -> torch.Tensor:
        return self.partition.threshold

    @property
    def good_clean_samples(self) -> torch.Tensor:
        return self.partition.good_clean_samples

    @property
    def bad_clean_samples(self) -> torch.Tensor:
        return self.partition.bad_clean_samples


@dataclass
class TrajectoryRecorder:
    """Utility to collect x_t states during a base diffusion rollout."""

    detach: bool = True
    clone: bool = True
    to_cpu: bool = False
    _latents_by_step: Dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def record(self, step: int, latents: torch.Tensor) -> None:
        step = int(step)
        if step in self._latents_by_step:
            raise ValueError(f"step {step} is already recorded")

        snapshot = latents.detach() if self.detach else latents
        if self.clone:
            snapshot = snapshot.clone()
        if self.to_cpu:
            snapshot = snapshot.cpu()
        self._latents_by_step[step] = snapshot

    def finalize(
        self,
        clean_samples: torch.Tensor,
        rewards: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        reward_inputs: Optional[torch.Tensor] = None,
    ) -> TrajectoryBatch:
        if not self._latents_by_step:
            raise ValueError("no steps were recorded")

        return TrajectoryBatch(
            latents_by_step=dict(self._latents_by_step),
            clean_samples=clean_samples,
            rewards=rewards,
            prompts=prompts,
            reward_inputs=reward_inputs,
        )
