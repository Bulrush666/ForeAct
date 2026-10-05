from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def resolve_config_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_path"]).parent / path
    return path.resolve()


def _require(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing required config key: {dotted_key}")
        value = value[key]
    return value


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "paths.python_bin",
        "paths.moge_source_dir",
        "paths.flex_gemm_source_dir",
        "paths.checkpoint",
        "paths.dataset_root",
        "paths.output_root",
        "data.camera",
        "data.horizon",
        "data.tail_policy",
        "preprocess.observation_size",
        "preprocess.model_size",
        "teacher.batch_size_per_gpu",
        "distributed.gpu_ids",
        "storage.episodes_per_directory",
    )
    for key in required:
        _require(config, key)

    horizon = int(config["data"]["horizon"])
    if horizon <= 0:
        raise ValueError("data.horizon must be positive")
    if config["data"]["tail_policy"] != "repeat_last":
        raise ValueError("Only data.tail_policy=repeat_last is supported")
    for key in ("observation_size", "model_size"):
        shape = config["preprocess"][key]
        if not isinstance(shape, list) or len(shape) != 2 or min(map(int, shape)) <= 0:
            raise ValueError(f"preprocess.{key} must be [height, width]")
    batch_size = int(config["teacher"]["batch_size_per_gpu"])
    if batch_size <= 0:
        raise ValueError("teacher.batch_size_per_gpu must be positive")
    gpu_ids = config["distributed"]["gpu_ids"]
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError("distributed.gpu_ids must be a non-empty list")


def target_frame_index(
    base_index: int,
    episode_length: int,
    horizon: int,
) -> int:
    """Return the padded chunk endpoint: min(t + horizon, last frame)."""
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if not 0 <= base_index < episode_length:
        raise IndexError(
            f"base_index={base_index} outside episode length {episode_length}"
        )
    return min(int(base_index) + int(horizon), int(episode_length) - 1)


def target_frame_indices(episode_length: int, horizon: int) -> list[int]:
    """Unique frames needed by all padded chunks in one episode."""
    if episode_length <= 0:
        return []
    if episode_length <= horizon:
        return [episode_length - 1]
    return list(range(horizon, episode_length))
