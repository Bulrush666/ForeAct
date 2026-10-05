#!/usr/bin/env python3
"""Precompute padded 16-step chunk-end MoGe-3 depth for Bridge/LeRobot.

This program is deliberately offline-only: every model/source/data path comes
from train.yaml and no from_pretrained repository name is accepted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from moge3_depth.config import (
    load_config,
    resolve_config_path,
    target_frame_indices,
)
from moge3_depth.storage import (
    FORMAT_VERSION,
    cache_path,
    load_episodes,
    validate_episode_cache,
    write_episode_cache_atomic,
    write_json_atomic,
)


LOGGER = logging.getLogger("moge3-precompute")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "train.yaml")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_logging(output_root: Path, shard_id: int) -> Path:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{shard_id:02d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def add_offline_sources(config: dict[str, Any]) -> None:
    source_paths = (
        resolve_config_path(config, config["paths"]["moge_source_dir"]),
        resolve_config_path(config, config["paths"]["flex_gemm_source_dir"]),
    )
    for path in reversed(source_paths):
        if not path.is_dir():
            raise FileNotFoundError(f"Required offline source directory: {path}")
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def video_path(dataset: Path, episode_index: int, camera: str) -> Path:
    return (
        dataset
        / "videos"
        / f"chunk-{episode_index // 1000:03d}"
        / f"observation.images.{camera}"
        / f"episode_{episode_index:06d}.mp4"
    )


def preprocess_image(
    rgb: np.ndarray,
    observation_size: tuple[int, int],
    model_size: tuple[int, int],
) -> torch.Tensor:
    """Match the VLA path: 256 RGB -> 224 square -> 320x576."""
    obs_h, obs_w = observation_size
    model_h, model_w = model_size
    image = Image.fromarray(rgb, mode="RGB").resize((obs_w, obs_h))
    tensor = (
        torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
    )
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(model_h, model_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0].clamp_(0, 1)


def infer_batch(
    model,
    tensors: list[torch.Tensor],
    *,
    device: torch.device,
    teacher_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = torch.stack(tensors).to(device=device, non_blocking=True)
    with torch.inference_mode():
        output = model.infer(
            images,
            resolution_level=int(teacher_cfg["resolution_level"]),
            refine_steps=int(teacher_cfg["refine_steps"]),
            use_fp16=bool(teacher_cfg["use_fp16"]),
            apply_mask=bool(teacher_cfg.get("apply_mask", True)),
        )
    depth = output["depth"].float().cpu().numpy()
    mask_output = output.get("mask")
    mask = (
        mask_output.bool().cpu().numpy()
        if mask_output is not None
        else np.ones_like(depth, dtype=np.bool_)
    )
    intrinsics = output["intrinsics"].float().cpu().numpy()
    return depth, mask, intrinsics


def load_reuse_episode_cache(
    path: Path,
    *,
    episode_index: int,
    episode_length: int,
    model_size: tuple[int, int],
    expected_horizon: int,
    camera: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reuse cache: {path}")

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

    with np.load(path, allow_pickle=False) as archive:
        missing_fields = required - set(archive.files)
        if missing_fields:
            raise ValueError(
                f"{path}: missing reuse fields {sorted(missing_fields)}"
            )
        if int(np.asarray(archive["format_version"]).item()) != FORMAT_VERSION:
            raise ValueError(f"{path}: format_version mismatch")
        if int(np.asarray(archive["episode_index"]).item()) != episode_index:
            raise ValueError(f"{path}: episode_index mismatch")
        if int(np.asarray(archive["episode_length"]).item()) != episode_length:
            raise ValueError(f"{path}: episode_length mismatch")
        if int(np.asarray(archive["horizon"]).item()) != expected_horizon:
            raise ValueError(
                f"{path}: expected reuse horizon {expected_horizon}, "
                f"got {int(np.asarray(archive['horizon']).item())}"
            )
        archived_camera = str(np.asarray(archive["camera"]).item())
        if archived_camera != camera:
            raise ValueError(
                f"{path}: camera {archived_camera!r} != {camera!r}"
            )

        depth = np.asarray(archive["depth"], dtype=np.float16).copy()
        valid = np.asarray(archive["valid_mask"], dtype=np.bool_).copy()
        available = np.asarray(archive["available"], dtype=np.bool_).copy()
        intrinsics = np.asarray(
            archive["intrinsics"], dtype=np.float32
        ).copy()

    expected_depth_shape = (episode_length, *map(int, model_size))
    if depth.shape != expected_depth_shape:
        raise ValueError(
            f"{path}: depth shape {depth.shape} != {expected_depth_shape}"
        )
    if valid.shape != expected_depth_shape:
        raise ValueError(f"{path}: valid_mask shape mismatch")
    if available.shape != (episode_length,):
        raise ValueError(f"{path}: available shape mismatch")
    if intrinsics.shape != (episode_length, 3, 3):
        raise ValueError(f"{path}: intrinsics shape mismatch")

    old_wanted = target_frame_indices(episode_length, expected_horizon)
    missing_old = [index for index in old_wanted if not available[index]]
    if missing_old:
        raise ValueError(
            f"{path}: old cache is incomplete; missing {missing_old[:8]}"
        )

    valid_on_available = valid & available[:, None, None]
    if not valid_on_available.any():
        raise ValueError(f"{path}: no valid reusable depth pixels")
    invalid_values = valid_on_available & (
        ~np.isfinite(depth) | (depth <= 0)
    )
    if invalid_values.any():
        raise ValueError(f"{path}: invalid reusable depth values")
    if not np.isfinite(intrinsics[available]).all():
        raise ValueError(f"{path}: non-finite reusable intrinsics")

    return depth, valid, available, intrinsics


def process_episode(
    model,
    *,
    dataset: Path,
    episode: dict[str, Any],
    camera: str,
    horizon: int,
    observation_size: tuple[int, int],
    model_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
    teacher_cfg: dict[str, Any],
    reuse_path: Path,
    reuse_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    episode_index = int(episode["episode_index"])
    episode_length = int(episode["length"])
    wanted = target_frame_indices(episode_length, horizon)

    depth_cache, valid_cache, available, intrinsics_cache = (
        load_reuse_episode_cache(
            reuse_path,
            episode_index=episode_index,
            episode_length=episode_length,
            model_size=model_size,
            expected_horizon=reuse_horizon,
            camera=camera,
        )
    )

    wanted_mask = np.zeros((episode_length,), dtype=np.bool_)
    wanted_mask[np.asarray(wanted, dtype=np.int64)] = True
    available &= wanted_mask
    depth_cache[~available] = 0
    valid_cache[~available] = False
    intrinsics_cache[~available] = 0

    wanted_set = {index for index in wanted if not available[index]}
    reused_frame_count = len(wanted) - len(wanted_set)
    inferred_frame_count = len(wanted_set)

    path = video_path(dataset, episode_index, camera)
    if wanted_set and not path.is_file():
        raise FileNotFoundError(path)

    pending_indices: list[int] = []
    pending_tensors: list[torch.Tensor] = []
    decoded_frames = 0
    inference_seconds = 0.0

    def flush() -> None:
        nonlocal inference_seconds
        if not pending_tensors:
            return
        started = time.perf_counter()
        depths, masks, intrinsics = infer_batch(
            model,
            pending_tensors,
            device=device,
            teacher_cfg=teacher_cfg,
        )
        inference_seconds += time.perf_counter() - started
        for local_index, frame_index in enumerate(pending_indices):
            depth = depths[local_index]
            valid = masks[local_index] & np.isfinite(depth) & (depth > 0)
            depth_cache[frame_index] = np.where(valid, depth, 0).astype(np.float16)
            valid_cache[frame_index] = valid
            available[frame_index] = True
            intrinsics_cache[frame_index] = intrinsics[local_index]
        pending_indices.clear()
        pending_tensors.clear()

    if wanted_set:
        last_needed = max(wanted_set)
        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                decoded_frames = frame_index + 1
                if frame_index not in wanted_set:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                pending_indices.append(frame_index)
                pending_tensors.append(
                    preprocess_image(rgb, observation_size, model_size)
                )
                if len(pending_tensors) >= batch_size:
                    flush()
                if frame_index >= last_needed:
                    break
        flush()

    missing = [index for index in wanted if not available[index]]
    if missing:
        raise RuntimeError(
            f"Episode {episode_index}: missing target frames {missing[:8]}; "
            f"decoded={decoded_frames}, metadata_length={episode_length}"
        )
    valid_values = depth_cache[valid_cache]
    if valid_values.size == 0:
        raise RuntimeError(f"Episode {episode_index}: MoGe returned no valid depth")
    stats = {
        "episode_index": episode_index,
        "episode_length": episode_length,
        "target_frame_count": len(wanted),
        "reused_frame_count": reused_frame_count,
        "inferred_frame_count": inferred_frame_count,
        "reuse_cache_path": str(reuse_path),
        "first_target_frame": wanted[0],
        "last_target_frame": wanted[-1],
        "decoded_frames": decoded_frames,
        "valid_ratio": float(valid_cache[available].mean()),
        "depth_min_m": float(valid_values.min()),
        "depth_median_m": float(np.median(valid_values)),
        "depth_max_m": float(valid_values.max()),
        "inference_seconds": inference_seconds,
    }
    return depth_cache, valid_cache, available, intrinsics_cache, stats


def ensure_run_metadata(
    config: dict[str, Any],
    output_root: Path,
    dataset: Path,
    checkpoint: Path,
) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "dataset": str(dataset),
        "dataset_name": config["data"].get("dataset_name", dataset.name),
        "camera": config["data"]["camera"],
        "horizon": int(config["data"]["horizon"]),
        "reuse_horizon": int(config["data"]["reuse_horizon"]),
        "reuse_cache_root": str(
            resolve_config_path(config, config["paths"]["reuse_cache_root"])
        ),
        "tail_policy": "repeat_last",
        "target_formula": "min(base_frame + horizon, episode_length - 1)",
        "checkpoint": str(checkpoint),
        "checkpoint_variant": config["teacher"]["variant"],
        "resolution_level": int(config["teacher"]["resolution_level"]),
        "refine_steps": int(config["teacher"]["refine_steps"]),
        "use_fp16": bool(config["teacher"]["use_fp16"]),
        "observation_size": list(map(int, config["preprocess"]["observation_size"])),
        "model_size": list(map(int, config["preprocess"]["model_size"])),
        "depth_unit": "meter",
        "depth_dtype": "float16",
        "invalid_depth_value": 0.0,
        "storage_layout": "cache/<dataset>/chunk-NNN/episode_NNNNNN.npz",
    }
    path = output_root / "metadata.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Existing metadata does not match train.yaml: {path}. "
                "Use a new output_root for a different cache contract."
            )
    else:
        try:
            write_json_atomic(path, payload)
        except FileExistsError:
            pass


def validate_reuse_run_metadata(
    config: dict[str, Any],
    reuse_root: Path,
    dataset: Path,
    checkpoint: Path,
    reuse_horizon: int,
) -> None:
    path = reuse_root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing reuse metadata: {path}")

    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format_version": FORMAT_VERSION,
        "dataset_name": config["data"].get(
            "dataset_name", dataset.name
        ),
        "camera": config["data"]["camera"],
        "horizon": reuse_horizon,
        "tail_policy": "repeat_last",
        "target_formula": (
            "min(base_frame + horizon, episode_length - 1)"
        ),
        "checkpoint_variant": config["teacher"]["variant"],
        "resolution_level": int(
            config["teacher"]["resolution_level"]
        ),
        "refine_steps": int(config["teacher"]["refine_steps"]),
        "use_fp16": bool(config["teacher"]["use_fp16"]),
        "observation_size": list(
            map(int, config["preprocess"]["observation_size"])
        ),
        "model_size": list(
            map(int, config["preprocess"]["model_size"])
        ),
        "depth_unit": "meter",
        "depth_dtype": "float16",
        "invalid_depth_value": 0.0,
    }

    mismatches = {}
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "actual": actual_value,
            }

    for key, expected_path in (
        ("dataset", dataset),
        ("checkpoint", checkpoint),
    ):
        actual_value = metadata.get(key)
        if actual_value is None:
            mismatches[key] = {
                "expected": str(expected_path),
                "actual": None,
            }
        elif (
            Path(str(actual_value)).expanduser().resolve()
            != expected_path.expanduser().resolve()
        ):
            mismatches[key] = {
                "expected": str(expected_path),
                "actual": actual_value,
            }

    if mismatches:
        raise ValueError(
            "Reuse metadata does not match current teacher/preprocess: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard-id < num-shards")

    config = load_config(args.config)
    paths = config["paths"]
    dataset = resolve_config_path(config, paths["dataset_root"])
    output_root = resolve_config_path(config, paths["output_root"])
    checkpoint = resolve_config_path(config, paths["checkpoint"])
    reuse_root = resolve_config_path(
        config, paths["reuse_cache_root"]
    )
    reuse_horizon = int(config["data"]["reuse_horizon"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(output_root, args.shard_id)
    LOGGER.info("config=%s shard=%d/%d device=%s", args.config, args.shard_id, args.num_shards, args.device)
    LOGGER.info("log=%s", log_path)

    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not reuse_root.is_dir():
        raise FileNotFoundError(reuse_root)
    # moge3_env inherits xFormers from foreact6, but that xFormers build is not
    # compatible with the Triton version required by pinned FlexGEMM. DINOv2
    # has a native PyTorch attention fallback which is verified for inference.
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    add_offline_sources(config)

    episodes = load_episodes(dataset / "meta" / "episodes.jsonl")
    episodes = [
        episode
        for episode in episodes
        if int(episode["episode_index"]) % args.num_shards == args.shard_id
    ]
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]
    LOGGER.info("assigned_episodes=%d", len(episodes))

    ensure_run_metadata(config, output_root, dataset, checkpoint)
    validate_reuse_run_metadata(
        config,
        reuse_root,
        dataset,
        checkpoint,
        reuse_horizon,
    )
    LOGGER.info(
        "reuse_root=%s reuse_horizon=%d",
        reuse_root,
        reuse_horizon,
    )
    if args.dry_run:
        LOGGER.info("dry-run complete; no model loaded and no cache written")
        return

    from moge.model.v3 import MoGeModel

    device = torch.device(args.device)
    random.seed(int(config["runtime"]["seed"]) + args.shard_id)
    torch.manual_seed(int(config["runtime"]["seed"]) + args.shard_id)
    load_started = time.perf_counter()
    model = MoGeModel.from_pretrained(str(checkpoint)).to(device).eval()
    model.requires_grad_(False)
    LOGGER.info("model_loaded_seconds=%.3f", time.perf_counter() - load_started)

    dataset_name = config["data"].get("dataset_name", dataset.name)
    camera = str(config["data"]["camera"])
    horizon = int(config["data"]["horizon"])
    observation_size = tuple(map(int, config["preprocess"]["observation_size"]))
    model_size = tuple(map(int, config["preprocess"]["model_size"]))
    batch_size = int(config["teacher"]["batch_size_per_gpu"])
    episodes_per_directory = int(config["storage"]["episodes_per_directory"])
    resume = bool(config["runtime"]["resume"])
    fail_fast = bool(config["runtime"]["fail_fast"])
    manifest_path = output_root / "manifests" / f"shard_{args.shard_id:02d}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    processed = skipped = failed = 0
    started = time.perf_counter()
    with manifest_path.open("a", encoding="utf-8", buffering=1) as manifest:
        for ordinal, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            destination = cache_path(
                output_root,
                dataset_name,
                episode_index,
                episodes_per_directory,
            )
            reuse_source = cache_path(
                reuse_root,
                dataset_name,
                episode_index,
                episodes_per_directory,
            )
            try:
                if destination.is_file() and resume:
                    stats = validate_episode_cache(
                        destination,
                        episode_index=episode_index,
                        episode_length=episode_length,
                        model_size=model_size,
                        horizon=horizon,
                    )
                    record = {"status": "skipped", "cache_path": str(destination), **stats}
                    skipped += 1
                else:
                    episode_started = time.perf_counter()
                    depth, valid, available, intrinsics, stats = process_episode(
                        model,
                        dataset=dataset,
                        episode=episode,
                        camera=camera,
                        horizon=horizon,
                        observation_size=observation_size,
                        model_size=model_size,
                        batch_size=batch_size,
                        device=device,
                        teacher_cfg=config["teacher"],
                        reuse_path=reuse_source,
                        reuse_horizon=reuse_horizon,
                    )
                    write_episode_cache_atomic(
                        destination,
                        depth=depth,
                        valid_mask=valid,
                        available=available,
                        intrinsics=intrinsics,
                        episode_index=episode_index,
                        horizon=horizon,
                        camera=camera,
                    )
                    record = {
                        "status": "processed",
                        "cache_path": str(destination),
                        "cache_size_bytes": destination.stat().st_size,
                        "episode_seconds": time.perf_counter() - episode_started,
                        **stats,
                    }
                    processed += 1
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                if ordinal == 1 or ordinal % int(config["runtime"]["log_every_episodes"]) == 0:
                    LOGGER.info(
                        "progress=%d/%d processed=%d skipped=%d failed=%d last_episode=%d",
                        ordinal,
                        len(episodes),
                        processed,
                        skipped,
                        failed,
                        episode_index,
                    )
            except Exception as error:
                failed += 1
                record = {
                    "status": "failed",
                    "episode_index": episode_index,
                    "episode_length": episode_length,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                LOGGER.exception("episode %d failed", episode_index)
                if fail_fast:
                    raise

    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "assigned_episodes": len(episodes),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output_root / "summaries" / f"shard_{args.shard_id:02d}.json", summary)
    LOGGER.info("summary=%s", json.dumps(summary))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

