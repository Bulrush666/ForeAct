#!/usr/bin/env python3
"""Precompute next-frame MoGe-3 depth for Fractal/RT-1 LeRobot v2.

This program is deliberately offline-only: every model/source/data path comes
from the YAML config and no remote model repository name is accepted.  Cache
frame ``k`` contains the teacher depth for real video frame ``k``; with
``horizon=1`` a training sample at frame ``t`` reads frame
``min(t + 1, episode_length - 1)``.
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


LOGGER = logging.getLogger("moge3-fractal-precompute")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "fractal_next1.yaml")
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


def validate_dataset_contract(dataset: Path, camera: str, horizon: int) -> dict[str, Any]:
    """Fail early if the configured directory is not the expected RT-1 dataset."""
    if horizon != 1:
        raise ValueError(
            f"Fractal next-frame generation requires data.horizon=1, got {horizon}"
        )
    info_path = dataset / "meta" / "info.json"
    episodes_path = dataset / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    if not episodes_path.is_file():
        raise FileNotFoundError(episodes_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2."):
        raise ValueError(f"Expected LeRobot v2.x, got {version!r}: {info_path}")
    camera_key = f"observation.images.{camera}"
    feature = info.get("features", {}).get(camera_key)
    if not isinstance(feature, dict) or feature.get("dtype") != "video":
        raise ValueError(f"Video feature {camera_key!r} is missing from {info_path}")
    return info


def preprocess_image(
    rgb: np.ndarray,
    observation_size: tuple[int, int],
    model_size: tuple[int, int],
) -> torch.Tensor:
    """Match the VLA path: source RGB -> 224 square -> 320x576."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    episode_index = int(episode["episode_index"])
    episode_length = int(episode["length"])
    wanted = target_frame_indices(episode_length, horizon)
    wanted_set = set(wanted)
    model_h, model_w = model_size
    depth_cache = np.zeros((episode_length, model_h, model_w), dtype=np.float16)
    valid_cache = np.zeros((episode_length, model_h, model_w), dtype=np.bool_)
    available = np.zeros((episode_length,), dtype=np.bool_)
    intrinsics_cache = np.zeros((episode_length, 3, 3), dtype=np.float32)

    path = video_path(dataset, episode_index, camera)
    if not path.is_file():
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
            if frame_index >= wanted[-1]:
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
                f"Existing metadata does not match the selected configuration: {path}. "
                "Use a new output_root for a different cache contract."
            )
    else:
        try:
            write_json_atomic(path, payload)
        except FileExistsError:
            pass


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard-id < num-shards")

    config = load_config(args.config)
    paths = config["paths"]
    dataset = resolve_config_path(config, paths["dataset_root"])
    output_root = resolve_config_path(config, paths["output_root"])
    checkpoint = resolve_config_path(config, paths["checkpoint"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(output_root, args.shard_id)
    LOGGER.info("config=%s shard=%d/%d device=%s", args.config, args.shard_id, args.num_shards, args.device)
    LOGGER.info("log=%s", log_path)

    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    camera = str(config["data"]["camera"])
    horizon = int(config["data"]["horizon"])
    dataset_info = validate_dataset_contract(dataset, camera, horizon)
    LOGGER.info(
        "dataset_version=%s episodes=%s frames=%s camera=observation.images.%s horizon=%d",
        dataset_info.get("codebase_version"),
        dataset_info.get("total_episodes"),
        dataset_info.get("total_frames"),
        camera,
        horizon,
    )
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
