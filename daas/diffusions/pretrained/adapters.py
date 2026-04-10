from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from daas.diffusions.evolution.schedules import DiffusionSchedule


@dataclass
class DiffusersPipelineAdapter:
    """
    Small compatibility wrapper around a diffusers pipeline.
    """

    pipeline: Any

    @property
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    @property
    def unet(self) -> Any:
        return self.pipeline.unet

    @property
    def vae(self) -> Any:
        return self.pipeline.vae

    @property
    def device(self) -> torch.device:
        return self.pipeline.device

    @property
    def latent_scaling_factor(self) -> float:
        vae_config = getattr(self.pipeline.vae, "config", None)
        return float(getattr(vae_config, "scaling_factor", 1.0))

    def make_schedule(self) -> DiffusionSchedule:
        return DiffusionSchedule.from_diffusers_scheduler(self.scheduler)

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None, **kwargs: Any) -> Any:
        self.scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        return self.scheduler.timesteps

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pt") -> Any:
        scaled_latents = latents / self.latent_scaling_factor
        decoded = self.pipeline.vae.decode(scaled_latents, return_dict=False)[0]
        if hasattr(self.pipeline, "image_processor"):
            return self.pipeline.image_processor.postprocess(decoded, output_type=output_type)
        return decoded

    def encode_images(self, images: torch.Tensor, sample: bool = False) -> torch.Tensor:
        posterior = self.pipeline.vae.encode(images).latent_dist
        latents = posterior.sample() if sample else posterior.mode()
        return latents * self.latent_scaling_factor
