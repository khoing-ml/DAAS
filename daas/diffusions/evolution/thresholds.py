from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


SplitResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _validate_rewards(rewards: torch.Tensor) -> None:
    if rewards.ndim != 1:
        raise ValueError("rewards must be a 1D tensor")


def _ensure_non_empty_good(rewards: torch.Tensor, good_mask: torch.Tensor) -> torch.Tensor:
    if good_mask.any():
        return good_mask
    fixed_mask = torch.zeros_like(good_mask, dtype=torch.bool)
    fixed_mask[torch.argmax(rewards)] = True
    return fixed_mask


@dataclass(frozen=True)
class FixedThreshold:
    value: float

    def split(self, rewards: torch.Tensor) -> SplitResult:
        _validate_rewards(rewards)
        threshold = rewards.new_tensor(float(self.value))
        good_mask = _ensure_non_empty_good(rewards, rewards >= threshold)
        return threshold, good_mask, ~good_mask


@dataclass(frozen=True)
class QuantileThreshold:
    quantile: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError("quantile must be between 0 and 1")

    def split(self, rewards: torch.Tensor) -> SplitResult:
        _validate_rewards(rewards)
        threshold = torch.quantile(rewards, self.quantile)
        good_mask = _ensure_non_empty_good(rewards, rewards >= threshold)
        return threshold, good_mask, ~good_mask


@dataclass(frozen=True)
class TopKThreshold:
    k: int = 1

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")

    def split(self, rewards: torch.Tensor) -> SplitResult:
        _validate_rewards(rewards)

        k = min(int(self.k), int(rewards.numel()))
        top_values, top_indices = torch.topk(rewards, k=k, largest=True, sorted=True)
        good_mask = torch.zeros_like(rewards, dtype=torch.bool)
        good_mask[top_indices] = True
        threshold = top_values[-1]
        return threshold, good_mask, ~good_mask


@dataclass(frozen=True)
class SecondBestThreshold:
    def split(self, rewards: torch.Tensor) -> SplitResult:
        _validate_rewards(rewards)
        k = 2 if rewards.numel() > 1 else 1
        return TopKThreshold(k=k).split(rewards)
