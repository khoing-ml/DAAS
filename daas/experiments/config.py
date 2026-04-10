from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from daas.diffusions.pretrained.sources import ModelSource, ModelSourceType
from daas.diffusions.pretrained.specs import PipelineSpec, PipelineTask


def _copy_mapping(data: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if data is None:
        return {}
    return dict(data)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, default_name: Optional[str] = None) -> "ComponentSpec":
        if "name" in data:
            name = str(data["name"])
        elif default_name is not None:
            name = default_name
        else:
            raise ValueError("component spec requires a `name` field")

        kwargs = _copy_mapping(data.get("kwargs"))
        for key, value in data.items():
            if key not in {"name", "kwargs"}:
                kwargs[key] = value
        return cls(name=name, kwargs=kwargs)


@dataclass(frozen=True)
class StepWindowConfig:
    min_step: Optional[int] = None
    max_step: Optional[int] = None
    min_fraction: Optional[float] = None
    max_fraction: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StepWindowConfig":
        return cls(
            min_step=data.get("min_step"),
            max_step=data.get("max_step"),
            min_fraction=data.get("min_fraction"),
            max_fraction=data.get("max_fraction"),
        )


@dataclass(frozen=True)
class ModelConfig:
    preset: Optional[str] = None
    source: Optional[ModelSource] = None
    task: PipelineTask = PipelineTask.TEXT_TO_IMAGE
    pipeline_class: Optional[str] = None
    scheduler_class: Optional[str] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)
    torch_dtype: Optional[Any] = None
    device: Optional[str] = None
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = False
    enable_xformers: bool = False
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelConfig":
        source = None
        source_data = data.get("source")
        if source_data is not None:
            source = ModelSource(
                location=str(source_data["location"]),
                source_type=ModelSourceType(str(source_data.get("type", ModelSourceType.HUGGINGFACE.value))),
                revision=source_data.get("revision"),
                variant=source_data.get("variant"),
                subfolder=source_data.get("subfolder"),
                use_safetensors=source_data.get("use_safetensors"),
                token=source_data.get("token"),
                local_files_only=bool(source_data.get("local_files_only", False)),
                extra_load_kwargs=_copy_mapping(source_data.get("extra_load_kwargs")),
            )

        task = PipelineTask(str(data.get("task", PipelineTask.TEXT_TO_IMAGE.value)))
        return cls(
            preset=data.get("preset"),
            source=source,
            task=task,
            pipeline_class=data.get("pipeline_class"),
            scheduler_class=data.get("scheduler_class"),
            scheduler_kwargs=_copy_mapping(data.get("scheduler_kwargs")),
            pipeline_kwargs=_copy_mapping(data.get("pipeline_kwargs")),
            torch_dtype=data.get("torch_dtype"),
            device=data.get("device"),
            enable_attention_slicing=bool(data.get("enable_attention_slicing", False)),
            enable_vae_slicing=bool(data.get("enable_vae_slicing", False)),
            enable_xformers=bool(data.get("enable_xformers", False)),
            enable_model_cpu_offload=bool(data.get("enable_model_cpu_offload", False)),
            enable_sequential_cpu_offload=bool(data.get("enable_sequential_cpu_offload", False)),
        )

    def to_pipeline_spec(self) -> PipelineSpec:
        if self.preset is None and self.source is None:
            raise ValueError("model config requires either `preset` or `source`")
        if self.source is None:
            raise ValueError("source must be resolved from a preset before building PipelineSpec")

        return PipelineSpec(
            source=self.source,
            task=self.task,
            pipeline_class=self.pipeline_class,
            scheduler_class=self.scheduler_class,
            scheduler_kwargs=dict(self.scheduler_kwargs),
            pipeline_kwargs=dict(self.pipeline_kwargs),
            torch_dtype=self.torch_dtype,
            device=self.device,
            enable_attention_slicing=self.enable_attention_slicing,
            enable_vae_slicing=self.enable_vae_slicing,
            enable_xformers=self.enable_xformers,
            enable_model_cpu_offload=self.enable_model_cpu_offload,
            enable_sequential_cpu_offload=self.enable_sequential_cpu_offload,
        )


@dataclass(frozen=True)
class RewardConfig:
    component: ComponentSpec

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RewardConfig":
        return cls(component=ComponentSpec.from_mapping(data))


@dataclass(frozen=True)
class SteeringConfig:
    guidance_scale: float = 1.0
    max_update_norm: Optional[float] = None
    threshold: ComponentSpec = field(default_factory=lambda: ComponentSpec(name="quantile", kwargs={"quantile": 0.75}))
    gate: ComponentSpec = field(default_factory=lambda: ComponentSpec(name="constant", kwargs={"scale": 1.0}))
    kernel: ComponentSpec = field(default_factory=lambda: ComponentSpec(name="rbf"))
    score_estimator: ComponentSpec = field(
        default_factory=lambda: ComponentSpec(name="good_set_mixture", kwargs={"temperature": 1.0})
    )
    step_window: Optional[StepWindowConfig] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SteeringConfig":
        threshold_data = data.get("threshold", {"name": "quantile", "quantile": 0.75})
        gate_data = data.get("gate", {"name": "constant", "scale": 1.0})
        kernel_data = data.get("kernel", {"name": "rbf"})
        estimator_data = data.get("score_estimator", {"name": "good_set_mixture", "temperature": 1.0})
        step_window_data = data.get("step_window")
        return cls(
            guidance_scale=float(data.get("guidance_scale", 1.0)),
            max_update_norm=data.get("max_update_norm"),
            threshold=ComponentSpec.from_mapping(threshold_data),
            gate=ComponentSpec.from_mapping(gate_data),
            kernel=ComponentSpec.from_mapping(kernel_data),
            score_estimator=ComponentSpec.from_mapping(estimator_data),
            step_window=StepWindowConfig.from_mapping(step_window_data) if step_window_data is not None else None,
        )


@dataclass(frozen=True)
class SamplingConfig:
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_particles: int = 4
    num_images_per_prompt: int = 1
    height: Optional[int] = None
    width: Optional[int] = None
    seed: Optional[int] = None
    output_dir: Optional[str] = None
    output_type: str = "pt"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SamplingConfig":
        known_fields = {
            "prompt",
            "negative_prompt",
            "num_inference_steps",
            "guidance_scale",
            "num_particles",
            "num_images_per_prompt",
            "height",
            "width",
            "seed",
            "output_dir",
            "output_type",
        }
        extra_kwargs = {key: value for key, value in data.items() if key not in known_fields}
        return cls(
            prompt=str(data["prompt"]),
            negative_prompt=data.get("negative_prompt"),
            num_inference_steps=int(data.get("num_inference_steps", 30)),
            guidance_scale=float(data.get("guidance_scale", 7.5)),
            num_particles=int(data.get("num_particles", 4)),
            num_images_per_prompt=int(data.get("num_images_per_prompt", 1)),
            height=data.get("height"),
            width=data.get("width"),
            seed=data.get("seed"),
            output_dir=data.get("output_dir"),
            output_type=str(data.get("output_type", "pt")),
            extra_kwargs=extra_kwargs,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    reward: RewardConfig
    steering: SteeringConfig
    sampling: SamplingConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            name=str(data["name"]),
            model=ModelConfig.from_mapping(data["model"]),
            reward=RewardConfig.from_mapping(data["reward"]),
            steering=SteeringConfig.from_mapping(data.get("steering", {})),
            sampling=SamplingConfig.from_mapping(data["sampling"]),
            metadata=_copy_mapping(data.get("metadata")),
        )
