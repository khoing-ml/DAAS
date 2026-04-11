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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading preset={args.preset} with reward_name={args.reward_name}...")
    print(f"Device: {device}, dtype: {torch_dtype}")
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

    print("Running image generation with trajectory recording...")
    recorder = TrajectoryRecorder(detach=True, clone=True, to_cpu=False)

    # Set up timesteps
    timesteps = bundle.adapter.set_timesteps(args.steps, device=device)
    guidance_scale = args.guidance_scale

    # Encode prompts
    batch_size = 1
    text_embeddings = bundle.pipeline.encode_prompts(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=args.negative_prompt,
    )

    # Initialize latents
    latents = torch.randn(
        (batch_size, 4, args.height // 8, args.width // 8),
        generator=generator,
        device=device,
        dtype=bundle.pipeline.dtype,
    )
    latents = latents * bundle.adapter.scheduler.init_noise_sigma

    print("Running reverse diffusion with trajectory recording...")
    for i, t in enumerate(timesteps):
        # Predict noise
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_model_input = bundle.adapter.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = bundle.pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Scheduler step
        latents = bundle.adapter.scheduler.step(noise_pred, t, latents).prev_sample

        # Record step
        recorder.record(int(t.item()), latents)

    print("Decoding latents to images...")
    images = bundle.adapter.decode_latents(latents, output_type="pil")
    if not isinstance(images, list):
        images = [images]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path)

    print("DAAS-guided SDXL smoke test completed")
    print(f"saved_image={out_path}")
    print(f"Used reward_name={args.reward_name}, steering_guidance_scale={args.steering_guidance_scale}")


if __name__ == "__main__":
    main()
