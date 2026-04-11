from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from daas.diffusions.evolution import (
    ConstantGate,
    DensityRatioGate,
    EvolutionSteerer,
    GoodSetScoreEstimator,
    KernelDensityScoreEstimator,
    QuantileThreshold,
    RBFKernel,
    StepWindow,
)
from daas.diffusions.pretrained import (
    DiffusersPipelineBundle,
    DiffusersPipelineLoader,
    PipelineTask,
    PipelineSpec,
    huggingface_source,
    make_preset,
)
from daas.experiments.rewards import build_reward_function


@dataclass
class SimpleInferenceComponents:
    pipeline_bundle: DiffusersPipelineBundle
    reward_fn: Any
    steerer: EvolutionSteerer


def load_huggingface_pipeline(
    model_id: str,
    *,
    task: str = "text-to-image",
    pipeline_class: Optional[str] = None,
    scheduler_class: Optional[str] = None,
    scheduler_kwargs: Optional[dict[str, Any]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
    torch_dtype: Any = None,
    device: Optional[str] = None,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
    subfolder: Optional[str] = None,
    use_safetensors: Optional[bool] = True,
    token: Optional[str] = None,
    local_files_only: bool = False,
    enable_attention_slicing: bool = False,
    enable_vae_slicing: bool = False,
    enable_xformers: bool = False,
    enable_model_cpu_offload: bool = False,
    enable_sequential_cpu_offload: bool = False,
) -> DiffusersPipelineBundle:
    normalized_task = task.strip().lower()
    task_mapping = {
        "text-to-image": PipelineTask.TEXT_TO_IMAGE,
        "image-to-image": PipelineTask.IMAGE_TO_IMAGE,
        "inpaint": PipelineTask.INPAINT,
    }
    if normalized_task not in task_mapping:
        raise KeyError(f"unsupported task: {task}")

    spec = PipelineSpec(
        source=huggingface_source(
            model_id,
            revision=revision,
            variant=variant,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            token=token,
            local_files_only=local_files_only,
        ),
        task=task_mapping[normalized_task],
        pipeline_class=pipeline_class,
        scheduler_class=scheduler_class,
        scheduler_kwargs=dict(scheduler_kwargs or {}),
        pipeline_kwargs=dict(pipeline_kwargs or {}),
        torch_dtype=torch_dtype,
        device=device,
        enable_attention_slicing=enable_attention_slicing,
        enable_vae_slicing=enable_vae_slicing,
        enable_xformers=enable_xformers,
        enable_model_cpu_offload=enable_model_cpu_offload,
        enable_sequential_cpu_offload=enable_sequential_cpu_offload,
    )
    return DiffusersPipelineLoader().load(spec)


def load_preset_pipeline(
    preset: str,
    *,
    torch_dtype: Any = None,
    device: Optional[str] = None,
    enable_attention_slicing: bool = False,
    enable_vae_slicing: bool = False,
    enable_xformers: bool = False,
    enable_model_cpu_offload: bool = False,
    enable_sequential_cpu_offload: bool = False,
    **overrides: Any,
) -> DiffusersPipelineBundle:
    spec = make_preset(
        preset,
        torch_dtype=torch_dtype,
        device=device,
        enable_attention_slicing=enable_attention_slicing,
        enable_vae_slicing=enable_vae_slicing,
        enable_xformers=enable_xformers,
        enable_model_cpu_offload=enable_model_cpu_offload,
        enable_sequential_cpu_offload=enable_sequential_cpu_offload,
        **overrides,
    )
    return DiffusersPipelineLoader().load(spec)


def build_simple_steerer(
    schedule: Any,
    *,
    guidance_scale: float = 1.0,
    threshold_quantile: float = 0.75,
    score_estimator: str = "mixture",
    estimator_temperature: float = 1.0,
    estimator_bandwidth: Optional[float] = None,
    estimator_chunk_size: Optional[int] = None,
    kernel_bandwidth: Optional[float] = None,
    gate: str = "constant",
    gate_temperature: float = 1.0,
    gate_bias: float = 0.0,
    gate_min_scale: float = 0.0,
    gate_max_scale: float = 1.0,
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
    max_update_norm: Optional[float] = None,
) -> EvolutionSteerer:
    normalized_estimator = score_estimator.strip().lower()
    if normalized_estimator in {"mixture", "good_set_mixture"}:
        estimator = GoodSetScoreEstimator(
            schedule=schedule,
            temperature=estimator_temperature,
            chunk_size=estimator_chunk_size,
        )
    elif normalized_estimator in {"kde", "kernel_density"}:
        estimator = KernelDensityScoreEstimator(
            schedule=schedule,
            bandwidth=estimator_bandwidth,
            chunk_size=estimator_chunk_size,
        )
    else:
        raise KeyError(f"unsupported score estimator: {score_estimator}")

    normalized_gate = gate.strip().lower()
    if normalized_gate == "constant":
        gate_module = ConstantGate(scale=1.0)
    elif normalized_gate == "density_ratio":
        gate_module = DensityRatioGate(
            temperature=gate_temperature,
            bias=gate_bias,
            min_scale=gate_min_scale,
            max_scale=gate_max_scale,
        )
    else:
        raise KeyError(f"unsupported gate: {gate}")

    step_window = None
    if min_step is not None or max_step is not None:
        step_window = StepWindow(min_step=min_step, max_step=max_step)

    return EvolutionSteerer(
        schedule=schedule,
        threshold_policy=QuantileThreshold(quantile=threshold_quantile),
        score_estimator=estimator,
        kernel=RBFKernel(bandwidth=kernel_bandwidth),
        gate=gate_module,
        guidance_scale=guidance_scale,
        step_window=step_window,
        max_update_norm=max_update_norm,
    )


def build_simple_inference_components(
    *,
    preset: Optional[str] = None,
    model_id: Optional[str] = None,
    reward_name: str = "pickscore",
    torch_dtype: Any = None,
    device: Optional[str] = None,
    guidance_scale: float = 1.0,
    threshold_quantile: float = 0.75,
    score_estimator: str = "mixture",
    estimator_temperature: float = 1.0,
    estimator_bandwidth: Optional[float] = None,
    estimator_chunk_size: Optional[int] = None,
    kernel_bandwidth: Optional[float] = None,
    gate: str = "constant",
    gate_temperature: float = 1.0,
    gate_bias: float = 0.0,
    gate_min_scale: float = 0.0,
    gate_max_scale: float = 1.0,
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
    max_update_norm: Optional[float] = None,
    **pipeline_overrides: Any,
) -> SimpleInferenceComponents:
    if (preset is None) == (model_id is None):
        raise ValueError("provide exactly one of `preset` or `model_id`")

    if preset is not None:
        bundle = load_preset_pipeline(
            preset,
            torch_dtype=torch_dtype,
            device=device,
            **pipeline_overrides,
        )
    else:
        bundle = load_huggingface_pipeline(
            model_id,
            torch_dtype=torch_dtype,
            device=device,
            **pipeline_overrides,
        )

    steerer = build_simple_steerer(
        bundle.schedule,
        guidance_scale=guidance_scale,
        threshold_quantile=threshold_quantile,
        score_estimator=score_estimator,
        estimator_temperature=estimator_temperature,
        estimator_bandwidth=estimator_bandwidth,
        estimator_chunk_size=estimator_chunk_size,
        kernel_bandwidth=kernel_bandwidth,
        gate=gate,
        gate_temperature=gate_temperature,
        gate_bias=gate_bias,
        gate_min_scale=gate_min_scale,
        gate_max_scale=gate_max_scale,
        min_step=min_step,
        max_step=max_step,
        max_update_norm=max_update_norm,
    )
    reward_fn = build_reward_function(
        reward_name,
        inference_dtype=torch_dtype,
        device=bundle.adapter.device,
    )
    return SimpleInferenceComponents(
        pipeline_bundle=bundle,
        reward_fn=reward_fn,
        steerer=steerer,
    )
