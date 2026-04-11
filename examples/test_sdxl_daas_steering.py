from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow `python examples/test_sdxl_daas_steering.py` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from daas.simple import build_simple_inference_components
from daas.diffusions.evolution import TrajectoryRecorder
from daas.experiments.logging import ExperimentRunLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDXL with DAAS evolution steering smoke test.")
    parser.add_argument("--preset", type=str, default="sdxl-base", help="Pipeline preset.")
    parser.add_argument("--prompt", type=str, default="cinematic photo of a futuristic city at sunrise")
    parser.add_argument("--negative-prompt", type=str, default="low quality, blurry, distorted")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto when omitted")
    parser.add_argument("--reward-name", type=str, default="pickscore", help="Reward model name.")
    parser.add_argument(
        "--steering-guidance-scale",
        type=float,
        default=0.75,
        help="DAAS steering guidance scale.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/sdxl_daas_steering.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for experiment logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize logger
    run_logger = ExperimentRunLogger(
        name="sdxl_daas_steering",
        log_dir=args.log_dir,
        run_name="smoke_test",
    )
    logger = run_logger.logger

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Log configuration
    config = {
        "preset": args.preset,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "device": device,
        "torch_dtype": str(torch_dtype),
        "reward_name": args.reward_name,
        "steering_guidance_scale": args.steering_guidance_scale,
    }
    run_logger.log_config(config)

    logger.info(f"Loading preset={args.preset} with reward_name={args.reward_name}...")
    logger.info(f"Device: {device}, dtype: {torch_dtype}")

    components = build_simple_inference_components(
        preset=args.preset,
        reward_name=args.reward_name,
        torch_dtype=torch_dtype,
        device=device,
        guidance_scale=args.steering_guidance_scale,
        enable_attention_slicing=True,
        enable_vae_slicing=True,
    )

    bundle = components.pipeline_bundle
    reward_fn = components.reward_fn
    steerer = components.steerer

    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    logger.info("Running image generation with trajectory recording...")
    recorder = TrajectoryRecorder(detach=True, clone=True, to_cpu=False)

    # Use pipeline's native __call__ for inference (handles prompts, latents, timesteps internally)
    logger.info("Running reverse diffusion with trajectory recording...")
    result = bundle.pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
        output_type="pil",
    )

    images = result.images
    
    # Log progress at end of generation
    run_logger.log_metrics(step=args.steps, generation_complete=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path)

    # Log summary
    summary = {
        "output_image": str(out_path),
        "run_dir": str(run_logger.run_dir),
        "status": "completed",
        "reward_name": args.reward_name,
        "steering_guidance_scale": args.steering_guidance_scale,
    }
    run_logger.write_summary(**summary)

    logger.info(f"DAAS-guided SDXL smoke test completed")
    logger.info(f"Output image: {out_path}")
    logger.info(f"Logs saved to: {run_logger.run_dir}")


if __name__ == "__main__":
    main()
