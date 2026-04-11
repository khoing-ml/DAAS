from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python examples/test_run.py` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from daas.diffusions.evolution import (
    DiffusionSchedule,
    EvolutionSteerer,
    QuantileThreshold,
    TrajectoryBatch,
)


def build_synthetic_batch(
    *,
    num_particles: int,
    channels: int,
    height: int,
    width: int,
    num_steps: int,
) -> TrajectoryBatch:
    shape = (num_particles, channels, height, width)
    clean_samples = torch.randn(shape)
    rewards = torch.linspace(0.0, 1.0, steps=num_particles)

    # Use simple noisy copies of clean samples for each diffusion step.
    latents_by_step = {
        step: clean_samples + 0.1 * torch.randn(shape)
        for step in range(num_steps)
    }

    return TrajectoryBatch(
        latents_by_step=latents_by_step,
        clean_samples=clean_samples,
        rewards=rewards,
    )


def run_smoke_test(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    schedule = DiffusionSchedule.linear(num_train_timesteps=args.num_steps)
    steerer = EvolutionSteerer(
        schedule=schedule,
        threshold_policy=QuantileThreshold(quantile=args.quantile),
        guidance_scale=args.guidance_scale,
        max_update_norm=args.max_update_norm,
    )

    trajectories = build_synthetic_batch(
        num_particles=args.num_particles,
        channels=args.channels,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
    )
    state = steerer.fit(trajectories)

    test_step = min(max(args.step, 0), args.num_steps - 1)
    latents = trajectories.latents_at(test_step)
    updated_latents, output = steerer.apply(latents, test_step, state)

    print("DAAS smoke test completed")
    print(f"step={output.step}, active={output.active}")
    print(f"good={state.partition.num_good}, bad={state.partition.num_bad}")
    print(f"delta_norm_mean={output.delta.flatten(start_dim=1).norm(dim=1).mean().item():.6f}")
    print(f"updated_shape={tuple(updated_latents.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight DAAS evolution smoke test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_smoke_test(parse_args())
