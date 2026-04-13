from __future__ import annotations

from typing import Any, Callable


def build_reward_function(name: str, *, inference_dtype: Any = None, device: Any = None, **kwargs: Any) -> Callable[..., Any]:
    normalized = name.strip().lower()

    if normalized == "jpeg_compressibility":
        from daas.reward_utils import jpeg_compressibility

        return jpeg_compressibility(inference_dtype=inference_dtype, device=device)

    if normalized == "clip_score":
        from daas.reward_utils import clip_score

        return clip_score(inference_dtype=inference_dtype, device=device, **kwargs)

    if normalized == "aesthetic_score":
        from daas.reward_utils import aesthetic_score

        return aesthetic_score(torch_dtype=inference_dtype, device=device, **kwargs)

    if normalized in {"pickscore", "picklescore"}:
        from daas.scorers.PickScore_scorer import PickScoreScorer

        scorer = PickScoreScorer(inference_dtype, device)
        scorer.requires_grad_(False)
        return scorer

    if normalized == "imagereward":
        try:
            from daas.scorers.ImageReward_scorer import ImageRewardScorer
        except ModuleNotFoundError as exc:
            if exc.name == "clip":
                raise ModuleNotFoundError(
                    "ImageReward requires the OpenAI CLIP package. Install it with: "
                    "pip install git+https://github.com/openai/CLIP.git"
                ) from exc
            raise

        scorer = ImageRewardScorer(inference_dtype, device)
        scorer.requires_grad_(False)
        return scorer

    raise KeyError(f"unknown reward function: {name}")
