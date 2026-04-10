from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from daas.diffusions.pretrained.sources import ModelSource


class PipelineTask(str, Enum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"
    INPAINT = "inpaint"


@dataclass(frozen=True)
class PipelineSpec:
    """
    Declarative config for loading a diffusers pipeline.
    """

    source: ModelSource
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

    def with_overrides(self, **updates: Any) -> "PipelineSpec":
        data = {
            "source": self.source,
            "task": self.task,
            "pipeline_class": self.pipeline_class,
            "scheduler_class": self.scheduler_class,
            "scheduler_kwargs": dict(self.scheduler_kwargs),
            "pipeline_kwargs": dict(self.pipeline_kwargs),
            "torch_dtype": self.torch_dtype,
            "device": self.device,
            "enable_attention_slicing": self.enable_attention_slicing,
            "enable_vae_slicing": self.enable_vae_slicing,
            "enable_xformers": self.enable_xformers,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
            "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload,
        }
        data.update(updates)
        return PipelineSpec(**data)
