from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sanitize_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.strip().lower())
    return safe.strip("-") or "run"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_run_logger(
    name: str,
    log_dir: str | Path = "logs",
    run_name: Optional[str] = None,
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """
    Create a per-run logger that writes both to stdout and to a file.
    """

    base_dir = Path(log_dir)
    resolved_run_name = _sanitize_name(run_name or name)
    run_dir = base_dir / f"{_timestamp()}-{resolved_run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    logger_name = f"{name}.{run_dir.name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, run_dir


class ExperimentRunLogger:
    """
    File-based experiment logger for inference runs and later statistics.
    """

    def __init__(
        self,
        name: str,
        log_dir: str | Path = "logs",
        run_name: Optional[str] = None,
        level: int = logging.INFO,
    ) -> None:
        self.logger, self.run_dir = create_run_logger(name=name, log_dir=log_dir, run_name=run_name, level=level)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"

    def log_config(self, config: Any) -> None:
        self.config_path.write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")
        self.logger.info("Saved config to %s", self.config_path)

    def log_message(self, message: str, *args: Any, level: int = logging.INFO) -> None:
        self.logger.log(level, message, *args)

    def log_metrics(self, step: Optional[int] = None, **metrics: Any) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "metrics": metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_json_default) + "\n")
        metric_summary = ", ".join(f"{key}={value}" for key, value in metrics.items())
        prefix = f"step={step} | " if step is not None else ""
        self.logger.info("%s%s", prefix, metric_summary)

    def log_reward_stats(self, rewards: torch.Tensor, prefix: str = "reward") -> Mapping[str, float]:
        if rewards.ndim != 1:
            raise ValueError("rewards must be a 1D tensor")

        stats = {
            f"{prefix}_mean": float(rewards.mean().item()),
            f"{prefix}_std": float(rewards.std(unbiased=False).item()),
            f"{prefix}_min": float(rewards.min().item()),
            f"{prefix}_max": float(rewards.max().item()),
        }
        self.log_metrics(**stats)
        return stats

    def write_summary(self, **summary: Any) -> None:
        self.summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
        self.logger.info("Saved summary to %s", self.summary_path)

