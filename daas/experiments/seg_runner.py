from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from daas.diffusions.evolution import TrajectoryBatch
from daas.experiments.builders import ExperimentComponents


@dataclass(frozen=True)
class SegRunResult:
    images: List[Any]
    rewards: torch.Tensor
    best_index: int
    best_reward: float
    loop_metrics: List[Dict[str, Any]]


class SegInferenceRunner:
    """
    Closed-loop latent population runner that applies evolution steering between
    generation loops.
    """

    def __init__(self, components: ExperimentComponents) -> None:
        self.components = components

    @staticmethod
    def _execution_device(pipeline: Any, fallback: torch.device) -> torch.device:
        maybe = getattr(pipeline, "_execution_device", None)
        if maybe is None:
            return fallback
        return torch.device(maybe)

    @staticmethod
    def _latent_shape(pipeline: Any, batch_size: int, height: int, width: int) -> tuple[int, int, int, int]:
        if height is None or width is None:
            raise ValueError("height and width are required for SEG latent population sampling")
        channels = int(getattr(pipeline.unet.config, "in_channels", 4))
        vae_scale = int(getattr(pipeline, "vae_scale_factor", 8))
        return (batch_size, channels, int(height) // vae_scale, int(width) // vae_scale)

    @staticmethod
    def _normalize_rewards(rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.detach()
        if rewards.ndim > 1:
            rewards = rewards.squeeze()
        if rewards.ndim != 1:
            raise ValueError("reward function must return a 1D tensor or squeeze-able tensor")
        return rewards

    def _reward_from_clean_latents(self, clean_latents: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        images_pt = self.components.pipeline_bundle.adapter.decode_latents(clean_latents, output_type="pt")
        if isinstance(images_pt, torch.Tensor):
            images_pt = images_pt.to(self.components.pipeline_bundle.adapter.device)
        with torch.no_grad():
            rewards = self.components.reward_fn(images=images_pt, prompts=list(prompts))
        return self._normalize_rewards(rewards)

    def _reward_from_latents(self, latents: torch.Tensor, prompts: Sequence[str], *, output_type: str = "pt") -> torch.Tensor:
        images = self.components.pipeline_bundle.adapter.decode_latents(latents, output_type=output_type)
        if isinstance(images, torch.Tensor):
            images = images.to(self.components.pipeline_bundle.adapter.device)
        with torch.no_grad():
            rewards = self.components.reward_fn(images=images, prompts=list(prompts))
        return self._normalize_rewards(rewards)

    def _pipeline_common_kwargs(self) -> Dict[str, Any]:
        sampling = self.components.config.sampling
        # These keys are controlled explicitly by the runner for latent-population inference.
        reserved_keys = {
            "prompt",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "latents",
            "num_images_per_prompt",
            "output_type",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        }
        passthrough_kwargs = {
            key: value
            for key, value in sampling.extra_kwargs.items()
            if not str(key).startswith("seg_") and key not in reserved_keys
        }
        return {
            "prompt": sampling.prompt,
            "negative_prompt": sampling.negative_prompt,
            "num_inference_steps": sampling.num_inference_steps,
            "guidance_scale": sampling.guidance_scale,
            "height": sampling.height,
            "width": sampling.width,
            "num_images_per_prompt": 1,
            **passthrough_kwargs,
        }

    def run(
        self,
        *,
        loops: int = 1,
        noise_scale: float = 0.0,
        elite_keep: int = 0,
        generator: torch.Generator | None = None,
        logger: Any = None,
    ) -> SegRunResult:
        if loops < 1:
            raise ValueError("loops must be >= 1")
        if noise_scale < 0:
            raise ValueError("noise_scale must be >= 0")
        if elite_keep < 0:
            raise ValueError("elite_keep must be >= 0")

        bundle = self.components.pipeline_bundle
        pipeline = bundle.pipeline
        steerer = self.components.steerer
        sampling = self.components.config.sampling

        batch_size = int(sampling.num_particles)
        if batch_size < 1:
            raise ValueError("sampling.num_particles must be >= 1")

        if generator is None:
            generator = torch.Generator(device="cpu")

        exec_device = self._execution_device(pipeline, bundle.adapter.device)
        common_kwargs = self._pipeline_common_kwargs()
        use_intermediate_rewards = bool(sampling.extra_kwargs.get("seg_use_intermediate_rewards", True))
        inner_stein_steps = max(1, int(sampling.extra_kwargs.get("seg_inner_stein_steps", 1)))
        ir_reward_output_type = str(sampling.extra_kwargs.get("seg_intermediate_reward_output_type", "pt"))

        latents_shape = self._latent_shape(
            pipeline=pipeline,
            batch_size=batch_size,
            height=sampling.height,
            width=sampling.width,
        )
        latent_dtype = getattr(pipeline.unet, "dtype", torch.float32)

        population = torch.randn(
            latents_shape,
            device=exec_device,
            dtype=latent_dtype,
            generator=generator,
        )

        timesteps = bundle.adapter.set_timesteps(sampling.num_inference_steps, device=exec_device)
        if isinstance(timesteps, torch.Tensor):
            step_values = [int(value.item()) for value in timesteps]
        else:
            step_values = [int(value) for value in timesteps]

        active_steps = [value for value in step_values if steerer.is_active(value)]
        if active_steps:
            control_step = active_steps[len(active_steps) // 2]
        else:
            control_step = step_values[len(step_values) // 2]

        prompts = [sampling.prompt] * batch_size
        negative_prompts = None
        if sampling.negative_prompt is not None:
            negative_prompts = [sampling.negative_prompt] * batch_size
        loop_metrics: List[Dict[str, Any]] = []
        final_rewards: torch.Tensor | None = None

        for loop_idx in range(loops):
            step_update_count = 0
            step_update_total_reward_delta = 0.0

            def _step_callback(_pipeline: Any, step_index: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
                nonlocal step_update_count
                nonlocal step_update_total_reward_delta

                latents = callback_kwargs.get("latents")
                if latents is None:
                    return callback_kwargs

                step_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
                if not steerer.is_active(step_value):
                    return callback_kwargs

                updated_latents = latents
                for inner_idx in range(inner_stein_steps):
                    rewards_before = self._reward_from_latents(
                        updated_latents,
                        prompts,
                        output_type=ir_reward_output_type,
                    )

                    trajectories = TrajectoryBatch(
                        latents_by_step={step_value: updated_latents.detach()},
                        clean_samples=updated_latents.detach(),
                        rewards=rewards_before.detach(),
                        prompts=prompts,
                    )
                    state = steerer.fit(trajectories)
                    steered_latents, steering_output = steerer.apply(updated_latents, step_value, state)

                    if noise_scale > 0 and torch.any(state.partition.bad_mask):
                        steered_latents = steered_latents.clone()
                        bad_mask = state.partition.bad_mask.to(device=steered_latents.device)
                        noise = noise_scale * torch.randn(
                            steered_latents[bad_mask].shape,
                            device=steered_latents.device,
                            dtype=steered_latents.dtype,
                            generator=generator,
                        )
                        steered_latents[bad_mask] = steered_latents[bad_mask] + noise

                    rewards_after = self._reward_from_latents(
                        steered_latents,
                        prompts,
                        output_type=ir_reward_output_type,
                    )
                    reward_delta = float((rewards_after.mean() - rewards_before.mean()).item())

                    if logger is not None:
                        logger.log_metrics(
                            step=loop_idx,
                            seg_loop=loop_idx,
                            seg_step_index=int(step_index),
                            seg_timestep=step_value,
                            seg_inner_step=inner_idx,
                            seg_threshold=float(state.threshold.item()),
                            seg_good_count=int(state.partition.num_good),
                            seg_bad_count=int(state.partition.num_bad),
                            seg_ir_reward_mean_before=float(rewards_before.mean().item()),
                            seg_ir_reward_mean_after=float(rewards_after.mean().item()),
                            seg_ir_reward_delta=reward_delta,
                            seg_steering_active=bool(steering_output.active),
                            seg_steering_delta_norm=float(
                                steering_output.delta.flatten(start_dim=1).norm(dim=1).mean().item()
                            ),
                            seg_steering_gate_mean=float(steering_output.gate.mean().item()),
                        )

                    updated_latents = steered_latents
                    step_update_count += 1
                    step_update_total_reward_delta += reward_delta

                callback_kwargs["latents"] = updated_latents
                return callback_kwargs

            run_kwargs: Dict[str, Any] = {
                **common_kwargs,
                "prompt": prompts,
                "negative_prompt": negative_prompts,
                "latents": population,
                "output_type": "latent",
            }

            for key in ("prompt_2", "negative_prompt_2"):
                value = run_kwargs.get(key)
                if isinstance(value, str):
                    run_kwargs[key] = [value] * batch_size
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    value_list = list(value)
                    if len(value_list) == 1 and batch_size > 1:
                        run_kwargs[key] = value_list * batch_size

            if use_intermediate_rewards:
                run_kwargs["callback_on_step_end"] = _step_callback
                run_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            try:
                result = pipeline(
                    **run_kwargs,
                )
            except TypeError:
                if use_intermediate_rewards:
                    # Older diffusers pipelines may not expose callback_on_step_end.
                    fallback_kwargs = {
                        **common_kwargs,
                        "prompt": prompts,
                        "negative_prompt": negative_prompts,
                        "latents": population,
                        "output_type": "latent",
                    }
                    result = pipeline(**fallback_kwargs)
                else:
                    raise
            clean_latents = result.images
            rewards = self._reward_from_clean_latents(clean_latents, prompts)
            final_rewards = rewards

            trajectories = TrajectoryBatch(
                latents_by_step={control_step: population.detach()},
                clean_samples=clean_latents.detach(),
                rewards=rewards.detach(),
                prompts=prompts,
            )
            state = steerer.fit(trajectories)

            good_count = int(state.partition.num_good)
            bad_count = int(state.partition.num_bad)
            threshold = float(state.threshold.item())

            loop_entry: Dict[str, Any] = {
                "loop": loop_idx,
                "reward_mean": float(rewards.mean().item()),
                "reward_max": float(rewards.max().item()),
                "reward_min": float(rewards.min().item()),
                "threshold": threshold,
                "good_count": good_count,
                "bad_count": bad_count,
                "intermediate_updates": int(step_update_count),
                "intermediate_reward_delta_total": float(step_update_total_reward_delta),
            }
            loop_metrics.append(loop_entry)
            if logger is not None:
                logger.log_metrics(step=loop_idx, **loop_entry)

            if loop_idx == loops - 1:
                break

            if use_intermediate_rewards:
                next_population = clean_latents
                steering_output = None
            else:
                updated_population, steering_output = steerer.apply(population, control_step, state)
                next_population = updated_population

            if elite_keep > 0:
                k = min(elite_keep, batch_size)
                elite_indices = torch.topk(rewards, k=k, largest=True).indices
                next_population = next_population.clone()
                next_population[elite_indices] = population[elite_indices]

            if noise_scale > 0 and not use_intermediate_rewards:
                bad_mask = state.partition.bad_mask.to(device=next_population.device)
                if torch.any(bad_mask):
                    next_population = next_population.clone()
                    noise = noise_scale * torch.randn(
                        next_population[bad_mask].shape,
                        device=next_population.device,
                        dtype=next_population.dtype,
                        generator=generator,
                    )
                    next_population[bad_mask] = next_population[bad_mask] + noise

            if logger is not None and steering_output is not None:
                logger.log_metrics(
                    step=loop_idx,
                    steering_active=bool(steering_output.active),
                    steering_delta_norm=float(steering_output.delta.flatten(start_dim=1).norm(dim=1).mean().item()),
                    steering_gate_mean=float(steering_output.gate.mean().item()),
                )

            population = next_population

        assert final_rewards is not None
        best_idx = int(torch.argmax(final_rewards).item())
        best_latent = population[best_idx : best_idx + 1]

        final_images = pipeline(
            **common_kwargs,
            latents=best_latent,
            output_type="pil",
        ).images

        return SegRunResult(
            images=final_images,
            rewards=final_rewards,
            best_index=best_idx,
            best_reward=float(final_rewards[best_idx].item()),
            loop_metrics=loop_metrics,
        )
