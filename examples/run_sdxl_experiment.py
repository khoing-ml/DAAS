#!/usr/bin/env python
"""
Run SDXL diffusion experiment with reward-guided steering.

This script loads an experiment configuration, builds the necessary components,
generates images, computes rewards, and logs results.

Usage:
    python run_sdxl_experiment.py --config config/experiments/sdxl_base_pickscore.toml
    python run_sdxl_experiment.py --config config/experiments/sdxl_base_pickscore_smoke_test.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import replace

import torch

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from daas.experiments.io import load_experiment_config
from daas.experiments.builders import ExperimentFactory
from daas.experiments.logging import ExperimentRunLogger
from daas.experiments.seg_runner import SegInferenceRunner


def resolve_device(config_device: str) -> str:
    """Resolve device configuration to actual device."""
    if config_device == "auto" or config_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif config_device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        return "cpu"
    return config_device


def resolve_dtype(torch_dtype: str, device: str) -> str:
    """Resolve dtype configuration, adjusting for device constraints."""
    if device == "cpu" and torch_dtype in {"float16", "fp16", "bfloat16", "bf16"}:
        print(f"WARNING: {torch_dtype} requested on CPU, using float32 instead")
        return "float32"
    return torch_dtype


def main(config_path: str) -> None:
    """Run SDXL experiment with given configuration."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Load config
    config = load_experiment_config(str(config_path))
    
    # Resolve device and dtype
    device = resolve_device(config.model.device)
    torch_dtype = resolve_dtype(config.model.torch_dtype, device)
    
    # Initialize logger
    log_config = config.metadata.get("log_config", {}) if hasattr(config, "metadata") else {}
    log_dir = config.sampling.output_dir or "logs"
    run_name = log_config.get("run_name", config.name)
    
    run_logger = ExperimentRunLogger(
        name=config.name,
        log_dir=log_dir,
        run_name=run_name,
    )
    logger = run_logger.logger
    
    # Log configuration
    config_dict = {
        "name": config.name,
        "model": {
            "preset": config.model.preset,
            "torch_dtype": str(config.model.torch_dtype),
            "device": device,
        },
        "reward": config.reward.component.name,
        "sampling": {
            "prompt": config.sampling.prompt,
            "num_inference_steps": config.sampling.num_inference_steps,
            "guidance_scale": config.sampling.guidance_scale,
            "height": config.sampling.height,
            "width": config.sampling.width,
            "seed": config.sampling.seed,
        },
    }
    run_logger.log_config(config_dict)
    
    logger.info(f"Loaded config: {config.name}")
    logger.info(f"Model preset: {config.model.preset}")
    logger.info(f"Device: {device} (requested: {config.model.device})")
    logger.info(f"Dtype: {torch_dtype} (requested: {config.model.torch_dtype})")
    logger.info(f"Reward: {config.reward.component.name}")
    
    # Build experiment components
    logger.info("Building experiment components...")
    factory = ExperimentFactory()
    
    # Override config device and dtype with resolved values
    resolved_config = replace(
        config,
        model=replace(
            config.model,
            device=device,
            torch_dtype=torch_dtype,
            enable_model_cpu_offload=False if device == "cpu" else config.model.enable_model_cpu_offload,
            enable_sequential_cpu_offload=False if device == "cpu" else config.model.enable_sequential_cpu_offload,
        ),
    )
    
    components = factory.build_experiment(resolved_config)
    
    logger.info(f"Running SEG closed-loop inference: {config.sampling.prompt}")
    
    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    if config.sampling.seed is not None:
        generator = generator.manual_seed(int(config.sampling.seed))

    seg_loops = int(config.sampling.extra_kwargs.get("seg_loops", 1))
    seg_noise_scale = float(config.sampling.extra_kwargs.get("seg_noise_scale", 0.0))
    seg_elite_keep = int(config.sampling.extra_kwargs.get("seg_elite_keep", 0))
    seg_use_intermediate_rewards = bool(config.sampling.extra_kwargs.get("seg_use_intermediate_rewards", True))
    seg_inner_stein_steps = int(config.sampling.extra_kwargs.get("seg_inner_stein_steps", 1))
    seg_intermediate_reward_output_type = str(
        config.sampling.extra_kwargs.get("seg_intermediate_reward_output_type", "pt")
    )

    logger.info(
        "SEG settings: loops=%s, elite_keep=%s, noise_scale=%s, use_ir=%s, inner_stein_steps=%s, ir_output_type=%s",
        seg_loops,
        seg_elite_keep,
        seg_noise_scale,
        seg_use_intermediate_rewards,
        seg_inner_stein_steps,
        seg_intermediate_reward_output_type,
    )

    runner = SegInferenceRunner(components)
    run_result = runner.run(
        loops=seg_loops,
        noise_scale=seg_noise_scale,
        elite_keep=seg_elite_keep,
        generator=generator,
        logger=run_logger,
    )

    reward_stats = run_logger.log_reward_stats(run_result.rewards, prefix="reward")
    logger.info(f"Final reward stats: {reward_stats}")
    logger.info(f"Best particle index: {run_result.best_index}")
    logger.info(f"Best reward: {run_result.best_reward:.6f}")

    # Save generated images
    output_dir = Path(log_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(run_result.images):
        img_path = output_dir / f"image_{idx:03d}.png"
        img.save(img_path)
        logger.info(f"Saved image: {img_path}")

    run_logger.write_summary(
        config=config.name,
        status="completed",
        best_index=run_result.best_index,
        best_reward=run_result.best_reward,
        reward_stats=dict(reward_stats),
        seg_loops=seg_loops,
        seg_elite_keep=seg_elite_keep,
        seg_noise_scale=seg_noise_scale,
        seg_use_intermediate_rewards=seg_use_intermediate_rewards,
        seg_inner_stein_steps=seg_inner_stein_steps,
        seg_intermediate_reward_output_type=seg_intermediate_reward_output_type,
        loop_metrics=run_result.loop_metrics,
    )
    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SDXL diffusion experiment with reward-guided steering."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiments/sdxl_base_pickscore.toml",
        help="Path to experiment config file (default: config/experiments/sdxl_base_pickscore.toml)",
    )
    
    args = parser.parse_args()
    main(args.config)
