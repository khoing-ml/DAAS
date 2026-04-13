from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from daas.diffusions import (
    DiffusersPipelineBundle,
    DiffusersPipelineLoader,
    EvolutionSteerer,
    StepWindow,
    huggingface_source,
    local_directory_source,
    make_preset,
    single_file_source,
)
from daas.experiments.config import (
    ComponentSpec,
    ExperimentConfig,
    ModelConfig,
    StepWindowConfig,
)
from daas.experiments.component_builders import (
    build_gate_component,
    build_kernel_component,
    build_score_estimator_component,
    build_threshold_component,
)
from daas.experiments.io import load_experiment_config
from daas.experiments.rewards import build_reward_function


@dataclass
class ExperimentComponents:
    config: ExperimentConfig
    pipeline_bundle: DiffusersPipelineBundle
    reward_fn: Any
    steerer: EvolutionSteerer


def _resolve_torch_dtype(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value

    normalized = value.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported torch dtype alias: {value}")
    return mapping[normalized]


class ExperimentFactory:
    def __init__(self, pipeline_loader: DiffusersPipelineLoader | None = None) -> None:
        self.pipeline_loader = pipeline_loader or DiffusersPipelineLoader()

    def build_model_config(self, config: ModelConfig) -> ModelConfig:
        torch_dtype = _resolve_torch_dtype(config.torch_dtype)
        if config.preset is not None:
            preset_spec = make_preset(config.preset, torch_dtype=torch_dtype, device=config.device)
            if config.source is not None:
                preset_spec = preset_spec.with_overrides(source=config.source)
            scheduler_kwargs = dict(preset_spec.scheduler_kwargs)
            scheduler_kwargs.update(config.scheduler_kwargs)
            pipeline_kwargs = dict(preset_spec.pipeline_kwargs)
            pipeline_kwargs.update(config.pipeline_kwargs)
            return ModelConfig(
                preset=config.preset,
                source=preset_spec.source,
                task=preset_spec.task,
                pipeline_class=config.pipeline_class or preset_spec.pipeline_class,
                scheduler_class=config.scheduler_class or preset_spec.scheduler_class,
                scheduler_kwargs=scheduler_kwargs,
                pipeline_kwargs=pipeline_kwargs,
                torch_dtype=torch_dtype,
                device=config.device,
                enable_attention_slicing=config.enable_attention_slicing,
                enable_vae_slicing=config.enable_vae_slicing,
                enable_xformers=config.enable_xformers,
                enable_model_cpu_offload=config.enable_model_cpu_offload,
                enable_sequential_cpu_offload=config.enable_sequential_cpu_offload,
            )

        if config.source is None:
            raise ValueError("model config requires either a preset or a source")

        source = config.source
        if source.source_type.value == "huggingface":
            source = huggingface_source(
                source.location,
                revision=source.revision,
                variant=source.variant,
                subfolder=source.subfolder,
                use_safetensors=source.use_safetensors,
                token=source.token,
                local_files_only=source.local_files_only,
                **source.extra_load_kwargs,
            )
        elif source.source_type.value == "local_directory":
            source = local_directory_source(
                source.location,
                revision=source.revision,
                variant=source.variant,
                subfolder=source.subfolder,
                use_safetensors=source.use_safetensors,
                local_files_only=source.local_files_only,
                **source.extra_load_kwargs,
            )
        elif source.source_type.value == "single_file":
            source = single_file_source(
                source.location,
                token=source.token,
                local_files_only=source.local_files_only,
                **source.extra_load_kwargs,
            )

        return ModelConfig(
            preset=config.preset,
            source=source,
            task=config.task,
            pipeline_class=config.pipeline_class,
            scheduler_class=config.scheduler_class,
            scheduler_kwargs=dict(config.scheduler_kwargs),
            pipeline_kwargs=dict(config.pipeline_kwargs),
            torch_dtype=torch_dtype,
            device=config.device,
            enable_attention_slicing=config.enable_attention_slicing,
            enable_vae_slicing=config.enable_vae_slicing,
            enable_xformers=config.enable_xformers,
            enable_model_cpu_offload=config.enable_model_cpu_offload,
            enable_sequential_cpu_offload=config.enable_sequential_cpu_offload,
        )

    def build_threshold(self, spec: ComponentSpec) -> Any:
        return build_threshold_component(spec.name, spec.kwargs)

    def build_gate(self, spec: ComponentSpec) -> Any:
        return build_gate_component(spec.name, spec.kwargs)

    def build_kernel(self, spec: ComponentSpec) -> Any:
        return build_kernel_component(spec.name, spec.kwargs)

    def build_score_estimator(self, spec: ComponentSpec, *, schedule: Any) -> Any:
        return build_score_estimator_component(spec.name, schedule=schedule, kwargs=spec.kwargs)

    def build_step_window(self, config: StepWindowConfig | None, *, schedule: Any) -> Any:
        if config is None:
            return None
        if config.min_fraction is not None or config.max_fraction is not None:
            return StepWindow.from_fractions(
                schedule,
                min_fraction=0.0 if config.min_fraction is None else float(config.min_fraction),
                max_fraction=1.0 if config.max_fraction is None else float(config.max_fraction),
            )
        return StepWindow(min_step=config.min_step, max_step=config.max_step)

    def build_reward(self, config: ExperimentConfig, *, inference_dtype: Any, device: Any) -> Any:
        return build_reward_function(
            config.reward.component.name,
            inference_dtype=inference_dtype,
            device=device,
            **config.reward.component.kwargs,
        )

    def build_experiment(self, config: ExperimentConfig) -> ExperimentComponents:
        resolved_model = self.build_model_config(config.model)
        pipeline_bundle = self.pipeline_loader.load(resolved_model.to_pipeline_spec())
        schedule = pipeline_bundle.schedule
        steerer = EvolutionSteerer(
            schedule=schedule,
            threshold_policy=self.build_threshold(config.steering.threshold),
            score_estimator=self.build_score_estimator(config.steering.score_estimator, schedule=schedule),
            kernel=self.build_kernel(config.steering.kernel),
            gate=self.build_gate(config.steering.gate),
            guidance_scale=config.steering.guidance_scale,
            step_window=self.build_step_window(config.steering.step_window, schedule=schedule),
            max_update_norm=config.steering.max_update_norm,
        )
        reward_fn = self.build_reward(
            config,
            inference_dtype=resolved_model.torch_dtype,
            device=pipeline_bundle.adapter.device,
        )
        return ExperimentComponents(
            config=config,
            pipeline_bundle=pipeline_bundle,
            reward_fn=reward_fn,
            steerer=steerer,
        )


def build_experiment_config(data: Mapping[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(data)


def build_experiment_components(path: str) -> ExperimentComponents:
    config = load_experiment_config(path)
    return ExperimentFactory().build_experiment(config)
