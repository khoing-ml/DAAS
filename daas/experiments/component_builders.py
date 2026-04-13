from __future__ import annotations

from typing import Any, Mapping

from daas.diffusions import (
    ConstantGate,
    DensityRatioGate,
    FixedThreshold,
    GoodSetScoreEstimator,
    KernelDensityScoreEstimator,
    QuantileThreshold,
    RBFKernel,
    SecondBestThreshold,
    TopKThreshold,
)


def _normalized_name(name: str) -> str:
    return str(name).strip().lower()


def build_threshold_component(name: str, kwargs: Mapping[str, Any] | None = None) -> Any:
    normalized = _normalized_name(name)
    resolved_kwargs = dict(kwargs or {})

    registry = {
        "fixed": lambda: FixedThreshold(**resolved_kwargs),
        "quantile": lambda: QuantileThreshold(**resolved_kwargs),
        "topk": lambda: TopKThreshold(**resolved_kwargs),
        "second_best": lambda: SecondBestThreshold(),
    }
    try:
        return registry[normalized]()
    except KeyError as exc:
        raise KeyError(f"unknown threshold component: {name}") from exc


def build_gate_component(name: str, kwargs: Mapping[str, Any] | None = None) -> Any:
    normalized = _normalized_name(name)
    resolved_kwargs = dict(kwargs or {})

    registry = {
        "constant": lambda: ConstantGate(**resolved_kwargs),
        "density_ratio": lambda: DensityRatioGate(**resolved_kwargs),
    }
    try:
        return registry[normalized]()
    except KeyError as exc:
        raise KeyError(f"unknown gate component: {name}") from exc


def build_kernel_component(name: str, kwargs: Mapping[str, Any] | None = None) -> Any:
    normalized = _normalized_name(name)
    resolved_kwargs = dict(kwargs or {})

    registry = {
        "rbf": lambda: RBFKernel(**resolved_kwargs),
    }
    try:
        return registry[normalized]()
    except KeyError as exc:
        raise KeyError(f"unknown kernel component: {name}") from exc


def build_score_estimator_component(name: str, *, schedule: Any, kwargs: Mapping[str, Any] | None = None) -> Any:
    normalized = _normalized_name(name)
    resolved_kwargs = dict(kwargs or {})

    registry = {
        "good_set_mixture": lambda: GoodSetScoreEstimator(schedule=schedule, **resolved_kwargs),
        "mixture": lambda: GoodSetScoreEstimator(schedule=schedule, **resolved_kwargs),
        "kernel_density": lambda: KernelDensityScoreEstimator(schedule=schedule, **resolved_kwargs),
        "kde": lambda: KernelDensityScoreEstimator(schedule=schedule, **resolved_kwargs),
    }
    try:
        return registry[normalized]()
    except KeyError as exc:
        raise KeyError(f"unknown score estimator component: {name}") from exc
