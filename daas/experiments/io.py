from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Dict

from daas.experiments.config import ExperimentConfig


def load_raw_config(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    content = file_path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(content)
    if suffix == ".toml":
        return tomllib.loads(content)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyYAML is required to load YAML experiment configs") from exc
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("experiment config root must be a mapping")
        return data

    raise ValueError(f"unsupported config format: {suffix}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(load_raw_config(path))
