from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow `python examples/test_sdxl_pipeline.py` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from daas.simple import load_preset_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDXL pipeline smoke test.")
    parser.add_argument("--preset", type=str, default="sdxl-base", help="Preset name in daas registry.")
    parser.add_argument("--prompt", type=str, default="cinematic photo of a futuristic city at sunrise")
    parser.add_argument("--negative-prompt", type=str, default="low quality, blurry, distorted")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto when omitted")
    parser.add_argument(
        "--out",
        type=str,
        default="assets/sdxl_smoke_test.png",
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

    print(f"Loading preset={args.preset} on device={device} dtype={torch_dtype}...")
    bundle = load_preset_pipeline(
        args.preset,
        torch_dtype=torch_dtype,
        device=device,
        enable_attention_slicing=True,
        enable_vae_slicing=True,
    )

    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    print("Running inference...")
    result = bundle.pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )

    image = result.images[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    print("SDXL smoke test completed")
    print(f"saved_image={out_path}")


if __name__ == "__main__":
    main()
