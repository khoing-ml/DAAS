from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from daas.diffusions.evolution.gating import ConstantGate
from daas.diffusions.evolution.kernels import RBFKernel
from daas.diffusions.evolution.score_estimators import GoodSetScoreEstimator, ScoreEstimator
from daas.diffusions.evolution.schedules import DiffusionSchedule
from daas.diffusions.evolution.thresholds import QuantileThreshold
from daas.diffusions.evolution.trajectories import EvolutionState, TrajectoryBatch


@dataclass(frozen=True)
class StepWindow:
    min_step: Optional[int] = None
    max_step: Optional[int] = None

    def contains(self, step: int) -> bool:
        step = int(step)
        if self.min_step is not None and step < self.min_step:
            return False
        if self.max_step is not None and step > self.max_step:
            return False
        return True

    @classmethod
    def from_fractions(
        cls,
        schedule: DiffusionSchedule,
        min_fraction: float = 0.2,
        max_fraction: float = 0.8,
    ) -> "StepWindow":
        if not 0.0 <= min_fraction <= 1.0 or not 0.0 <= max_fraction <= 1.0:
            raise ValueError("fractions must be between 0 and 1")
        low, high = sorted((min_fraction, max_fraction))
        max_index = schedule.num_steps - 1
        return cls(
            min_step=int(round(low * max_index)),
            max_step=int(round(high * max_index)),
        )


@dataclass(frozen=True)
class SteeringOutput:
    step: int
    active: bool
    delta: torch.Tensor
    target_score: torch.Tensor
    vector_field: torch.Tensor
    gate: torch.Tensor


class EvolutionSteerer:
    """
    High-level controller for the good-vs-bad evolution steering loop.
    """

    def __init__(
        self,
        schedule: DiffusionSchedule,
        threshold_policy: Optional[object] = None,
        score_estimator: Optional[ScoreEstimator] = None,
        kernel: Optional[RBFKernel] = None,
        gate: Optional[object] = None,
        guidance_scale: float = 1.0,
        step_window: Optional[StepWindow] = None,
        max_update_norm: Optional[float] = None,
    ) -> None:
        if guidance_scale < 0:
            raise ValueError("guidance_scale must be >= 0")
        if max_update_norm is not None and max_update_norm <= 0:
            raise ValueError("max_update_norm must be > 0")

        self.schedule = schedule
        self.threshold_policy = threshold_policy or QuantileThreshold(quantile=0.75)
        self.score_estimator = score_estimator or GoodSetScoreEstimator(schedule=schedule)
        self.kernel = kernel or RBFKernel()
        self.gate = gate or ConstantGate(scale=1.0)
        self.guidance_scale = float(guidance_scale)
        self.step_window = step_window
        self.max_update_norm = max_update_norm

    def fit(self, trajectories: TrajectoryBatch) -> EvolutionState:
        return EvolutionState(partition=trajectories.partition(self.threshold_policy))

    def is_active(self, step: int) -> bool:
        if self.step_window is None:
            return True
        return self.step_window.contains(step)

    def _zero_output(self, latents: torch.Tensor, step: int) -> SteeringOutput:
        zeros = torch.zeros_like(latents)
        gate = latents.new_zeros((latents.shape[0],))
        return SteeringOutput(
            step=int(step),
            active=False,
            delta=zeros,
            target_score=zeros,
            vector_field=zeros,
            gate=gate,
        )

    def _broadcast_gate(self, gate: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        view_shape = (gate.shape[0],) + (1,) * (latents.ndim - 1)
        return gate.view(view_shape)

    def _clip_update_norm(self, delta: torch.Tensor) -> torch.Tensor:
        if self.max_update_norm is None:
            return delta

        flat = delta.flatten(start_dim=1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scales = (self.max_update_norm / norms).clamp(max=1.0)
        view_shape = (delta.shape[0],) + (1,) * (delta.ndim - 1)
        return delta * scales.view(view_shape)

    def steer(self, latents: torch.Tensor, step: int, state: EvolutionState) -> SteeringOutput:
        step = int(step)
        if not self.is_active(step):
            return self._zero_output(latents, step)

        target_score = self.score_estimator.score(latents, step, state.good_clean_samples)
        vector_field = self.kernel.vector_field(latents, target_score)
        gate = self.gate(
            latents,
            step,
            self.score_estimator,
            state.good_clean_samples,
            state.bad_clean_samples,
        ).to(device=latents.device, dtype=latents.dtype)

        delta = self.guidance_scale * self._broadcast_gate(gate, latents) * vector_field
        delta = self._clip_update_norm(delta)

        return SteeringOutput(
            step=step,
            active=True,
            delta=delta,
            target_score=target_score,
            vector_field=vector_field,
            gate=gate,
        )

    def apply(self, latents: torch.Tensor, step: int, state: EvolutionState) -> Tuple[torch.Tensor, SteeringOutput]:
        output = self.steer(latents, step, state)
        return latents + output.delta, output
