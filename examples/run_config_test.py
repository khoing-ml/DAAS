from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from daas.experiments.io import load_experiment_config
from daas.experiments.builders import ExperimentFactory
from daas.experiments.logging import ExperimentRunLogger
import numpy as np


def main() -> None:
    config_path = REPO_ROOT / "config" / "experiments" / "sdxl_base_pickscore_smoke_test.toml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Load config
    config = load_experiment_config(str(config_path))
    
    # Resolve device: auto -> cuda if available, else cpu
    device = config.model.device
    if device == "auto" or device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger_msg = "WARNING: CUDA requested but not available, falling back to CPU"
        print(logger_msg)
        device = "cpu"
    
    # Adjust dtype for CPU (use float32 if device is cpu)
    torch_dtype = config.model.torch_dtype
    if device == "cpu" and torch_dtype in {"float16", "fp16", "bfloat16", "bf16"}:
        dtype_msg = f"WARNING: {torch_dtype} requested on CPU, using float32 instead"
        print(dtype_msg)
        torch_dtype = "float32"
    
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
    
    # Log config
    config_dict = {
        "name": config.name,
        "model": {
            "preset": config.model.preset,
            "torch_dtype": str(config.model.torch_dtype),
            "device": config.model.device,
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
    from dataclasses import replace
    resolved_config = replace(
        config,
        model=replace(config.model, device=device, torch_dtype=torch_dtype)
    )
    
    components = factory.build_experiment(resolved_config)
    
    bundle = components.pipeline_bundle
    reward_fn = components.reward_fn
    steerer = components.steerer
    
    logger.info(f"Running inference: {config.sampling.prompt}")
    
    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(config.sampling.seed)
    
    # Generate image
    result = bundle.pipeline(
        prompt=config.sampling.prompt,
        negative_prompt=config.sampling.negative_prompt,
        num_inference_steps=config.sampling.num_inference_steps,
        guidance_scale=config.sampling.guidance_scale,
        height=config.sampling.height,
        width=config.sampling.width,
        generator=generator,
        output_type="pil",
    )
    
    images = result.images
    logger.info(f"Generated {len(images)} image(s)")
    
    # Compute rewards
    logger.info("Computing reward scores...")
    reward_stats = None
    try:
        image_tensors = []
        for pil_img in images:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img_array = torch.from_numpy(
                np.array(pil_img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            image_tensors.append(img_array)
        
        image_batch = torch.cat(image_tensors, dim=0).to(device)
        
        with torch.no_grad():
            rewards = reward_fn(
                images=image_batch,
                prompts=[config.sampling.prompt] * len(images),
            )
        
        if rewards.ndim > 1:
            rewards = rewards.squeeze()
        
        reward_stats = run_logger.log_reward_stats(rewards, prefix="reward")
        logger.info(f"Reward stats: {reward_stats}")
    except Exception as e:
        logger.warning(f"Failed to compute reward scores: {e}")
    
    # Save image
    output_dir = Path(config.sampling.output_dir or "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.name}_output.png"
    images[0].save(output_path)
    logger.info(f"Saved image: {output_path}")
    
    # Write summary
    summary = {
        "config": config.name,
        "output_image": str(output_path),
        "run_dir": str(run_logger.run_dir),
        "status": "completed",
        "reward_name": config.reward.component.name,
    }
    
    if reward_stats is not None:
        summary.update(reward_stats)
    
    run_logger.write_summary(**summary)
    logger.info(f"Logs saved to: {run_logger.run_dir}")


if __name__ == "__main__":
    main()
