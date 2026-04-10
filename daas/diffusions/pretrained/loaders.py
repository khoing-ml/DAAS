from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from daas.diffusions.pretrained.adapters import DiffusersPipelineAdapter
from daas.diffusions.pretrained.sources import ModelSourceType
from daas.diffusions.pretrained.specs import PipelineSpec, PipelineTask


@dataclass
class DiffusersPipelineBundle:
    spec: PipelineSpec
    pipeline: Any
    adapter: DiffusersPipelineAdapter

    @property
    def schedule(self):  # pragma: no cover - thin convenience wrapper
        return self.adapter.make_schedule()


class DiffusersPipelineLoader:
    """
    Loads pretrained diffusers pipelines from Hugging Face repos, local directories,
    or single-file checkpoints.
    """

    PIPELINE_CLASS_BY_TASK = {
        PipelineTask.TEXT_TO_IMAGE: "AutoPipelineForText2Image",
        PipelineTask.IMAGE_TO_IMAGE: "AutoPipelineForImage2Image",
        PipelineTask.INPAINT: "AutoPipelineForInpainting",
    }

    def resolve_pipeline_class_name(self, spec: PipelineSpec) -> str:
        return spec.pipeline_class or self.PIPELINE_CLASS_BY_TASK[spec.task]

    def _resolve_diffusers_class(self, class_name: str) -> type:
        try:
            import diffusers
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise ImportError(
                "diffusers is required to load pretrained pipelines. Install diffusers to use this module."
            ) from exc

        if not hasattr(diffusers, class_name):
            raise ValueError(f"diffusers does not expose class `{class_name}`")
        return getattr(diffusers, class_name)

    def _load_pipeline(self, spec: PipelineSpec) -> Any:
        pipeline_class = self._resolve_diffusers_class(self.resolve_pipeline_class_name(spec))
        load_kwargs = spec.source.to_diffusers_kwargs()
        load_kwargs.update(spec.pipeline_kwargs)
        if spec.torch_dtype is not None:
            load_kwargs["torch_dtype"] = spec.torch_dtype

        if spec.source.source_type == ModelSourceType.SINGLE_FILE:
            pipeline = pipeline_class.from_single_file(spec.source.location, **load_kwargs)
        else:
            pipeline = pipeline_class.from_pretrained(spec.source.location, **load_kwargs)
        return pipeline

    def _apply_scheduler_override(self, pipeline: Any, spec: PipelineSpec) -> None:
        if spec.scheduler_class is None:
            return
        scheduler_class = self._resolve_diffusers_class(spec.scheduler_class)
        config = pipeline.scheduler.config
        pipeline.scheduler = scheduler_class.from_config(config, **spec.scheduler_kwargs)

    def _apply_runtime_options(self, pipeline: Any, spec: PipelineSpec) -> Any:
        if spec.enable_attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if spec.enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if spec.enable_xformers and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            pipeline.enable_xformers_memory_efficient_attention()

        if spec.enable_sequential_cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif spec.enable_model_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        elif spec.device is not None:
            pipeline = pipeline.to(spec.device)

        return pipeline

    def load(self, spec: PipelineSpec) -> DiffusersPipelineBundle:
        pipeline = self._load_pipeline(spec)
        self._apply_scheduler_override(pipeline, spec)
        pipeline = self._apply_runtime_options(pipeline, spec)
        adapter = DiffusersPipelineAdapter(pipeline=pipeline)
        return DiffusersPipelineBundle(spec=spec, pipeline=pipeline, adapter=adapter)

    def load_preset(self, name: str, **overrides: Any) -> DiffusersPipelineBundle:
        from daas.diffusions.pretrained.registry import make_preset

        return self.load(make_preset(name, **overrides))
