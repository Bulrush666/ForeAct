from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_VERSION = 2


def load_episodes(path: str | Path) -> list[dict[str, Any]]:
    episodes = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def cache_path(
    output_root: str | Path,
    dataset_name: str,
    episode_index: int,
    episodes_per_directory: int = 1000,
) -> Path:
    episode_index = int(episode_index)
    directory_index = episode_index // int(episodes_per_directory)
    safe_name = str(dataset_name).replace("/", "_").replace("\\", "_")
    return (
        Path(output_root)
        / "cache"
        / safe_name
        / f"chunk-{directory_index:03d}"
        / f"episode_{episode_index:06d}.npz"
    )


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_episode_cache_atomic(
    path: str | Path,
    *,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    available: np.ndarray,
    intrinsics: np.ndarray,
    episode_index: int,
    horizon: int,
    camera: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.int32(FORMAT_VERSION),
            depth=np.asarray(depth, dtype=np.float16),
            valid_mask=np.asarray(valid_mask, dtype=np.bool_),
            available=np.asarray(available, dtype=np.bool_),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
            episode_index=np.int64(episode_index),
            episode_length=np.int64(depth.shape[0]),
            horizon=np.int64(horizon),
            camera=np.asarray(camera),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_episode_cache(
    path: str | Path,
    *,
    episode_index: int,
    episode_length: int,
    model_size: tuple[int, int],
    horizon: int,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "format_version",
            "depth",
            "valid_mask",
            "available",
            "intrinsics",
            "episode_index",
            "episode_length",
            "horizon",
            "camera",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        depth = archive["depth"]
        valid = archive["valid_mask"]
        available = archive["available"]
        intrinsics = archive["intrinsics"]
        expected_depth_shape = (int(episode_length), *map(int, model_size))
        if depth.shape != expected_depth_shape:
            raise ValueError(
                f"{path}: depth shape {depth.shape} != {expected_depth_shape}"
            )
        if valid.shape != depth.shape:
            raise ValueError(f"{path}: valid_mask shape does not match depth")
        if available.shape != (int(episode_length),):
            raise ValueError(f"{path}: invalid available shape {available.shape}")
        if intrinsics.shape != (int(episode_length), 3, 3):
            raise ValueError(f"{path}: invalid intrinsics shape {intrinsics.shape}")
        if int(archive["episode_index"]) != int(episode_index):
            raise ValueError(f"{path}: episode_index mismatch")
        if int(archive["episode_length"]) != int(episode_length):
            raise ValueError(f"{path}: episode_length mismatch")
        if int(archive["horizon"]) != int(horizon):
            raise ValueError(f"{path}: horizon mismatch")
        inferred = np.flatnonzero(available)
        if inferred.size == 0:
            raise ValueError(f"{path}: no available depth frames")
        valid_values = depth[valid]
        if valid_values.size == 0:
            raise ValueError(f"{path}: no valid positive depth pixels")
        if not np.isfinite(valid_values).all() or not (valid_values > 0).all():
            raise ValueError(f"{path}: invalid depth values under valid_mask")
        return {
            "episode_index": int(episode_index),
            "episode_length": int(episode_length),
            "available_frames": int(available.sum()),
            "first_available_frame": int(inferred[0]),
            "last_available_frame": int(inferred[-1]),
            "valid_ratio_available": float(valid[available].mean()),
            "depth_min_m": float(valid_values.min()),
            "depth_median_m": float(np.median(valid_values)),
            "depth_max_m": float(valid_values.max()),
            "size_bytes": path.stat().st_size,
        }
