from __future__ import annotations

from typing import Dict

from daas.diffusions.pretrained.sources import ModelSource, huggingface_source
from daas.diffusions.pretrained.specs import PipelineSpec, PipelineTask


def sdxl_base_text2image(source: ModelSource | None = None, **overrides: object) -> PipelineSpec:
    spec = PipelineSpec(
        source=source or huggingface_source("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True),
        task=PipelineTask.TEXT_TO_IMAGE,
        pipeline_class="StableDiffusionXLPipeline",
    )
    return spec.with_overrides(**overrides)


def sdxl_turbo_text2image(source: ModelSource | None = None, **overrides: object) -> PipelineSpec:
    spec = PipelineSpec(
        source=source or huggingface_source("stabilityai/sdxl-turbo", use_safetensors=True),
        task=PipelineTask.TEXT_TO_IMAGE,
        pipeline_class="StableDiffusionXLPipeline",
    )
    return spec.with_overrides(**overrides)


def sd15_text2image(source: ModelSource | None = None, **overrides: object) -> PipelineSpec:
    spec = PipelineSpec(
        source=source or huggingface_source("runwayml/stable-diffusion-v1-5", use_safetensors=True),
        task=PipelineTask.TEXT_TO_IMAGE,
        pipeline_class="StableDiffusionPipeline",
    )
    return spec.with_overrides(**overrides)


PRESET_SPECS: Dict[str, PipelineSpec] = {
    "sdxl-base": sdxl_base_text2image(),
    "sdxl-turbo": sdxl_turbo_text2image(),
    "sd15": sd15_text2image(),
}


def make_preset(name: str, source: ModelSource | None = None, **overrides: object) -> PipelineSpec:
    normalized = name.strip().lower()
    if normalized == "sdxl-base":
        return sdxl_base_text2image(source=source, **overrides)
    if normalized == "sdxl-turbo":
        return sdxl_turbo_text2image(source=source, **overrides)
    if normalized == "sd15":
        return sd15_text2image(source=source, **overrides)
    raise KeyError(f"unknown pretrained preset: {name}")
