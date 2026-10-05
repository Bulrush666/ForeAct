"""MoGe-3 episode-cache loading for chunk-end depth supervision."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def depth_cache_path(
    cache_root: str | Path,
    dataset_name: str,
    episode_index: int,
    episodes_per_directory: int = 1000,
) -> Path:
    """Match moge3_code's cache/<dataset>/chunk-NNN/episode_NNNNNN.npz layout."""
    episode_index = int(episode_index)
    safe_name = str(dataset_name).replace("/", "_").replace("\\", "_")
    return (
        Path(cache_root)
        / "cache"
        / safe_name
        / f"chunk-{episode_index // int(episodes_per_directory):03d}"
        / f"episode_{episode_index:06d}.npz"
    )


def metric_depth_to_normalized_log(
    depth: torch.Tensor, min_depth_m: float, max_depth_m: float
) -> torch.Tensor:
    log_min = float(np.log(min_depth_m))
    log_max = float(np.log(max_depth_m))
    clipped = depth.float().clamp(min=min_depth_m, max=max_depth_m)
    return 2 * (clipped.log() - log_min) / (log_max - log_min) - 1


def metric_depth_to_genception_target(
    depth: torch.Tensor, normalization_median_m: float, log_alpha: float
) -> torch.Tensor:
    if not np.isfinite(normalization_median_m) or normalization_median_m <= 0:
        raise ValueError("normalization median must be finite and positive")
    relative_depth = depth.float().clamp_min(0) / normalization_median_m
    target_01 = (float(log_alpha) * torch.log1p(relative_depth)).clamp(0, 1)
    return target_01.mul(2).sub(1)


def _resize_depth_and_mask(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    model_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(depth.shape[-2:]) == tuple(model_size):
        return depth, valid_mask
    depth = F.interpolate(
        depth.unsqueeze(1), size=model_size, mode="bilinear",
        align_corners=False, antialias=True,
    )[:, 0]
    valid_mask = F.interpolate(
        valid_mask.float().unsqueeze(1), size=model_size, mode="nearest"
    )[:, 0].bool()
    return depth, valid_mask


class MoGeDepthCacheReader:
    """Read metric MoGe depth with a per-worker episode LRU."""

    def __init__(
        self,
        cache_root: str | Path | None,
        *,
        dataset_name: str,
        frame_offsets: tuple[int, ...] = (16,),
        expected_horizon: int | None = None,
        require_depth: bool = False,
        cache_size: int = 4,
        episodes_per_directory: int = 1000,
        model_size: tuple[int, int] = (320, 576),
        min_depth_m: float = 0.05,
        max_depth_m: float = 10.0,
        normalization: str = "genception_median_log",
        median_scope: str = "frame",
        log_alpha: float = 0.8613366,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser() if cache_root else None
        self.dataset_name = str(dataset_name)
        self.frame_offsets = tuple(int(value) for value in frame_offsets)
        self.expected_horizon = (
            int(expected_horizon) if expected_horizon is not None else None
        )
        self.require_depth = bool(require_depth)
        self.cache_size = max(int(cache_size), 1)
        self.episodes_per_directory = int(episodes_per_directory)
        self.model_size = tuple(int(value) for value in model_size)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.normalization = str(normalization)
        self.median_scope = str(median_scope)
        self.log_alpha = float(log_alpha)
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._episode_medians: dict[Path, float] = {}
        if self.require_depth and self.cache_root is None:
            raise ValueError("require_depth=True requires depth_cache_root")
        if self.expected_horizon is not None and self.expected_horizon <= 0:
            raise ValueError("expected_horizon must be positive")
        if self.normalization not in {"genception_median_log", "metric_log"}:
            raise ValueError(f"Unsupported depth normalization: {self.normalization}")
        if self.median_scope not in {"frame", "episode"}:
            raise ValueError("depth_median_scope must be frame or episode")

    def _load(self, path: Path) -> dict[str, np.ndarray] | None:
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            required = {"depth", "valid_mask", "available", "episode_index", "horizon"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Depth cache {path} is missing {sorted(missing)}")
            payload = {name: archive[name] for name in archive.files}
        stored_horizon = int(np.asarray(payload["horizon"]).item())
        if (
            self.expected_horizon is not None
            and stored_horizon != self.expected_horizon
        ):
            raise ValueError(
                f"Depth cache horizon mismatch in {path}: "
                f"expected {self.expected_horizon}, got {stored_horizon}"
            )
        self._cache[path] = payload
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def load_sample(
        self, episode_index: int, frame_index: int, episode_length: int
    ) -> dict:
        if self.cache_root is None:
            return {}
        path = depth_cache_path(
            self.cache_root, self.dataset_name, episode_index,
            self.episodes_per_directory,
        )
        cache = self._load(path)
        if cache is None:
            if self.require_depth:
                raise FileNotFoundError(f"Missing MoGe depth cache: {path}")
            return {}
        if int(cache["episode_index"]) != int(episode_index):
            raise ValueError(f"episode_index mismatch in {path}")

        depth = np.asarray(cache["depth"])
        valid_mask = np.asarray(cache["valid_mask"], dtype=np.bool_)
        available = np.asarray(cache["available"], dtype=np.bool_)
        if depth.ndim != 3 or valid_mask.shape != depth.shape:
            raise ValueError(f"Depth cache arrays must be matching [T,H,W]: {path}")
        usable_frames = min(int(episode_length), depth.shape[0])
        if usable_frames <= 0 or available.shape[0] < usable_frames:
            raise ValueError(f"Invalid episode length/available array in {path}")

        raw_indices = np.asarray(self.frame_offsets, dtype=np.int64) + int(frame_index)
        # moge3_code contract: padded chunks reuse the last real frame.
        indices = np.clip(raw_indices, 0, usable_frames - 1)
        if not available[indices].all():
            missing = indices[~available[indices]].tolist()
            message = f"Unavailable MoGe target frames {missing} in {path}"
            if self.require_depth:
                raise ValueError(message)
            return {}

        episode_median = self._episode_medians.get(path)
        if episode_median is None:
            episode_valid = valid_mask[:usable_frames].copy()
            episode_valid &= available[:usable_frames, None, None]
            episode_valid &= np.isfinite(depth[:usable_frames])
            episode_valid &= depth[:usable_frames] >= self.min_depth_m
            episode_valid &= depth[:usable_frames] <= self.max_depth_m
            if not episode_valid.any():
                raise ValueError(f"No valid metric depth in {path}")
            episode_median = float(np.median(depth[:usable_frames][episode_valid]))
            self._episode_medians[path] = episode_median

        selected_depth = torch.from_numpy(depth[indices].astype(np.float32))
        selected_valid = torch.from_numpy(valid_mask[indices].copy())
        selected_depth, selected_valid = _resize_depth_and_mask(
            selected_depth, selected_valid, self.model_size
        )
        selected_valid &= torch.isfinite(selected_depth)
        selected_valid &= selected_depth >= self.min_depth_m
        selected_valid &= selected_depth <= self.max_depth_m

        medians = torch.full((len(indices),), episode_median, dtype=torch.float32)
        if self.median_scope == "frame":
            for item in range(len(indices)):
                values = selected_depth[item][selected_valid[item]]
                if values.numel():
                    medians[item] = torch.quantile(values.float(), 0.5)

        normalized_frames = []
        for item, median in enumerate(medians):
            median_value = float(median)
            filled = torch.where(
                selected_valid[item], selected_depth[item],
                torch.full_like(selected_depth[item], median_value),
            )
            if self.normalization == "genception_median_log":
                target = metric_depth_to_genception_target(
                    filled, median_value, self.log_alpha
                )
            else:
                target = metric_depth_to_normalized_log(
                    filled, self.min_depth_m, self.max_depth_m
                )
            normalized_frames.append(target)

        normalized = torch.stack(normalized_frames)
        return {
            "depth_target": normalized.unsqueeze(0).contiguous(),
            "depth_valid_mask": selected_valid.unsqueeze(0).contiguous(),
            "depth_frame_mask": torch.ones(len(indices), dtype=torch.bool),
            "depth_target_is_padded": torch.from_numpy(raw_indices >= usable_frames),
            "depth_target_indices": torch.from_numpy(indices.copy()),
            "depth_scene_median_m": episode_median,
            "depth_normalization_median_m": medians.contiguous(),
            "depth_cache_path": str(path),
        }
