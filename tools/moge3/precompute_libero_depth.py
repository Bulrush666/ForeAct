#!/usr/bin/env python3
"""Precompute next-frame MoGe-3 depth for LIBERO in LeRobot v3.0 format.

LeRobot v3 stores many episodes in each physical video file. This worker
therefore shards and decodes by physical MP4 instead of reopening an AV1 file
for every episode. Cache files remain episode-scoped and match the existing
WorldVLA MoGe cache contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

from moge3_depth.config import load_config, resolve_config_path, target_frame_indices
from moge3_depth.storage import (
    FORMAT_VERSION,
    cache_path,
    validate_episode_cache,
    write_episode_cache_atomic,
    write_json_atomic,
)


LOGGER = logging.getLogger("moge3-libero-precompute")


@dataclass(frozen=True)
class EpisodeDescriptor:
    suite_name: str
    suite_root: Path
    episode_index: int
    length: int
    fps: float
    video_chunk_index: int
    video_file_index: int
    start_frame: int
    stop_frame: int
    video_path: Path


@dataclass(frozen=True)
class VideoJob:
    suite_name: str
    video_chunk_index: int
    video_file_index: int
    video_path: Path
    episodes: tuple[EpisodeDescriptor, ...]

    @property
    def target_frames(self) -> int:
        return sum(max(1, item.length - 1) for item in self.episodes)

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.suite_name, self.video_chunk_index, self.video_file_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "libero_next1.yaml")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--episode-index", type=int, default=None)
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


def preprocess_image(
    rgb: np.ndarray,
    observation_size: tuple[int, int],
    model_size: tuple[int, int],
) -> torch.Tensor:
    """Match the VLA path: square observation resize, then Cosmos/MoGe resize."""
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


def _episode_metadata_paths(suite_root: Path) -> list[Path]:
    paths = sorted((suite_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata under {suite_root}")
    return paths


def load_suite_episodes(suite_root: Path, camera_key: str) -> list[EpisodeDescriptor]:
    info_path = suite_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if str(info.get("codebase_version")) != "v3.0":
        raise ValueError(f"{suite_root}: expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    if camera_key not in info.get("features", {}):
        raise KeyError(f"{suite_root}: camera {camera_key!r} is absent from meta/info.json")

    fps = float(info["fps"])
    prefix = f"videos/{camera_key}"
    columns = [
        "episode_index",
        "length",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    ]
    rows: list[dict[str, Any]] = []
    for metadata_path in _episode_metadata_paths(suite_root):
        rows.extend(pq.read_table(metadata_path, columns=columns).to_pylist())

    episodes: list[EpisodeDescriptor] = []
    video_template = str(info["video_path"])
    for row in rows:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        chunk_index = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        start_frame = int(round(float(row[f"{prefix}/from_timestamp"]) * fps))
        metadata_stop = int(round(float(row[f"{prefix}/to_timestamp"]) * fps))
        stop_frame = start_frame + length
        if length <= 0:
            raise ValueError(f"{suite_root}: episode {episode_index} has length {length}")
        if metadata_stop != stop_frame:
            raise ValueError(
                f"{suite_root}: episode {episode_index} timestamp span gives "
                f"stop={metadata_stop}, but start+length={stop_frame}"
            )
        relative_video = video_template.format(
            video_key=camera_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        path = suite_root / relative_video
        if not path.is_file():
            raise FileNotFoundError(path)
        episodes.append(
            EpisodeDescriptor(
                suite_name=suite_root.name,
                suite_root=suite_root,
                episode_index=episode_index,
                length=length,
                fps=fps,
                video_chunk_index=chunk_index,
                video_file_index=file_index,
                start_frame=start_frame,
                stop_frame=stop_frame,
                video_path=path,
            )
        )

    episodes.sort(key=lambda item: item.episode_index)
    indices = [item.episode_index for item in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{suite_root}: duplicate episode indices")
    expected_total = int(info["total_episodes"])
    if len(episodes) != expected_total:
        raise ValueError(f"{suite_root}: loaded {len(episodes)} episodes, expected {expected_total}")
    return episodes


def discover_video_jobs(config: dict[str, Any]) -> list[VideoJob]:
    dataset_root = resolve_config_path(config, config["paths"]["dataset_root"])
    camera_key = str(config["data"]["camera"])
    suite_names = config["data"].get("suites")
    if not isinstance(suite_names, list) or not suite_names:
        raise ValueError("data.suites must list one or more LIBERO suite directories")

    groups: dict[tuple[str, int, int], list[EpisodeDescriptor]] = {}
    for suite_name in suite_names:
        suite_root = dataset_root / str(suite_name)
        if not suite_root.is_dir():
            raise FileNotFoundError(suite_root)
        for episode in load_suite_episodes(suite_root, camera_key):
            key = (
                episode.suite_name,
                episode.video_chunk_index,
                episode.video_file_index,
            )
            groups.setdefault(key, []).append(episode)

    jobs: list[VideoJob] = []
    for key, episodes in groups.items():
        episodes.sort(key=lambda item: item.start_frame)
        if episodes[0].start_frame != 0:
            raise ValueError(f"{key}: first episode starts at frame {episodes[0].start_frame}")
        for previous, current in zip(episodes, episodes[1:]):
            if previous.stop_frame != current.start_frame:
                raise ValueError(
                    f"{key}: non-contiguous episodes {previous.episode_index} and "
                    f"{current.episode_index} ({previous.stop_frame} != {current.start_frame})"
                )
        jobs.append(
            VideoJob(
                suite_name=key[0],
                video_chunk_index=key[1],
                video_file_index=key[2],
                video_path=episodes[0].video_path,
                episodes=tuple(episodes),
            )
        )
    return sorted(jobs, key=lambda item: item.key)


def filter_jobs(
    jobs: Iterable[VideoJob],
    *,
    suite: str | None,
    episode_index: int | None,
) -> list[VideoJob]:
    if episode_index is not None and suite is None:
        raise ValueError("--episode-index requires --suite because indices repeat across suites")
    filtered: list[VideoJob] = []
    for job in jobs:
        if suite is not None and job.suite_name != suite:
            continue
        episodes = tuple(
            item
            for item in job.episodes
            if episode_index is None or item.episode_index == episode_index
        )
        if episodes:
            filtered.append(
                VideoJob(
                    suite_name=job.suite_name,
                    video_chunk_index=job.video_chunk_index,
                    video_file_index=job.video_file_index,
                    video_path=job.video_path,
                    episodes=episodes,
                )
            )
    if suite is not None and not filtered:
        raise ValueError(f"No matching episodes for suite={suite!r}, episode={episode_index}")
    return filtered


def assign_jobs_greedily(jobs: Iterable[VideoJob], num_shards: int) -> list[list[VideoJob]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    assignments: list[list[VideoJob]] = [[] for _ in range(num_shards)]
    loads = [0 for _ in range(num_shards)]
    for job in sorted(jobs, key=lambda item: (-item.target_frames, item.key)):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        assignments[shard].append(job)
        loads[shard] += job.target_frames
    for shard_jobs in assignments:
        shard_jobs.sort(key=lambda item: item.key)
    return assignments


def limit_job_episodes(jobs: Iterable[VideoJob], limit: int | None) -> list[VideoJob]:
    if limit is None:
        return list(jobs)
    if limit < 0:
        raise ValueError("--limit-episodes must be non-negative")
    remaining = limit
    output: list[VideoJob] = []
    for job in jobs:
        if remaining <= 0:
            break
        episodes = job.episodes[:remaining]
        remaining -= len(episodes)
        if episodes:
            output.append(
                VideoJob(
                    suite_name=job.suite_name,
                    video_chunk_index=job.video_chunk_index,
                    video_file_index=job.video_file_index,
                    video_path=job.video_path,
                    episodes=episodes,
                )
            )
    return output


def ensure_run_metadata(
    config: dict[str, Any],
    output_root: Path,
    checkpoint: Path,
    jobs: list[VideoJob],
) -> None:
    all_episodes = [episode for job in jobs for episode in job.episodes]
    payload = {
        "format_version": FORMAT_VERSION,
        "dataset_format": "lerobot_v3.0",
        "dataset_root": str(resolve_config_path(config, config["paths"]["dataset_root"])),
        "suites": list(config["data"]["suites"]),
        "camera": str(config["data"]["camera"]),
        "horizon": int(config["data"]["horizon"]),
        "tail_policy": "repeat_last",
        "target_formula": "min(base_frame + horizon, episode_length - 1)",
        "episode_count": len(all_episodes),
        "source_frame_count": sum(item.length for item in all_episodes),
        "target_frame_count": sum(max(1, item.length - 1) for item in all_episodes),
        "physical_video_count": len(jobs),
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
        "storage_layout": "cache/<suite>/chunk-NNN/episode_NNNNNN.npz",
    }
    path = output_root / "metadata.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Existing metadata does not match {config['_config_path']}: {path}. "
                "Use a new output_root for a different cache contract."
            )
    else:
        write_json_atomic(path, payload)


def process_video_job(
    model,
    job: VideoJob,
    *,
    output_root: Path,
    horizon: int,
    observation_size: tuple[int, int],
    model_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
    teacher_cfg: dict[str, Any],
    episodes_per_directory: int,
    record_callback: Callable[[dict[str, Any]], None],
) -> None:
    episodes = sorted(job.episodes, key=lambda item: item.start_frame)
    if not episodes:
        return

    cursor = 0
    active: EpisodeDescriptor | None = None
    wanted: list[int] = []
    wanted_set: set[int] = set()
    depth_cache: np.ndarray | None = None
    valid_cache: np.ndarray | None = None
    available: np.ndarray | None = None
    intrinsics_cache: np.ndarray | None = None
    pending_indices: list[int] = []
    pending_tensors: list[torch.Tensor] = []
    inference_seconds = 0.0
    episode_started = 0.0

    def start_episode(episode: EpisodeDescriptor) -> None:
        nonlocal active, wanted, wanted_set, depth_cache, valid_cache, available
        nonlocal intrinsics_cache, inference_seconds, episode_started
        active = episode
        wanted = target_frame_indices(episode.length, horizon)
        wanted_set = set(wanted)
        model_h, model_w = model_size
        depth_cache = np.zeros((episode.length, model_h, model_w), dtype=np.float16)
        valid_cache = np.zeros((episode.length, model_h, model_w), dtype=np.bool_)
        available = np.zeros((episode.length,), dtype=np.bool_)
        intrinsics_cache = np.zeros((episode.length, 3, 3), dtype=np.float32)
        inference_seconds = 0.0
        episode_started = time.perf_counter()

    def flush() -> None:
        nonlocal inference_seconds
        if not pending_tensors:
            return
        assert depth_cache is not None
        assert valid_cache is not None
        assert available is not None
        assert intrinsics_cache is not None
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

    def finish_episode() -> None:
        assert active is not None
        assert depth_cache is not None
        assert valid_cache is not None
        assert available is not None
        assert intrinsics_cache is not None
        missing = [index for index in wanted if not available[index]]
        if missing:
            raise RuntimeError(
                f"{active.suite_name} episode {active.episode_index}: "
                f"missing target frames {missing[:8]}"
            )
        valid_values = depth_cache[valid_cache]
        if valid_values.size == 0:
            raise RuntimeError(
                f"{active.suite_name} episode {active.episode_index}: no valid MoGe depth"
            )
        destination = cache_path(
            output_root,
            active.suite_name,
            active.episode_index,
            episodes_per_directory,
        )
        write_episode_cache_atomic(
            destination,
            depth=depth_cache,
            valid_mask=valid_cache,
            available=available,
            intrinsics=intrinsics_cache,
            episode_index=active.episode_index,
            horizon=horizon,
            camera=CONFIG_CAMERA_KEY,
        )
        record_callback(
            {
                "status": "processed",
                "suite": active.suite_name,
                "episode_index": active.episode_index,
                "episode_length": active.length,
                "target_frame_count": len(wanted),
                "first_target_frame": wanted[0],
                "last_target_frame": wanted[-1],
                "valid_ratio": float(valid_cache[available].mean()),
                "depth_min_m": float(valid_values.min()),
                "depth_median_m": float(np.median(valid_values)),
                "depth_max_m": float(valid_values.max()),
                "inference_seconds": inference_seconds,
                "episode_seconds": time.perf_counter() - episode_started,
                "cache_path": str(destination),
                "cache_size_bytes": destination.stat().st_size,
                "source_video": str(job.video_path),
            }
        )

    last_required_frame = episodes[-1].stop_frame - 1
    with av.open(str(job.video_path)) as container:
        stream = container.streams.video[0]
        if stream.frames and int(stream.frames) < episodes[-1].stop_frame:
            raise RuntimeError(
                f"{job.video_path}: container reports {stream.frames} frames, "
                f"but metadata needs {episodes[-1].stop_frame}"
            )
        for global_frame_index, frame in enumerate(container.decode(video=0)):
            if cursor >= len(episodes):
                break
            episode = episodes[cursor]
            if global_frame_index < episode.start_frame:
                continue
            if global_frame_index >= episode.stop_frame:
                raise RuntimeError(
                    f"{job.video_path}: skipped episode boundary for {episode.episode_index}"
                )
            if active is None:
                if global_frame_index != episode.start_frame:
                    raise RuntimeError(
                        f"{job.video_path}: expected episode {episode.episode_index} at "
                        f"frame {episode.start_frame}, decoded {global_frame_index}"
                    )
                start_episode(episode)

            local_frame_index = global_frame_index - episode.start_frame
            if local_frame_index in wanted_set:
                rgb = frame.to_ndarray(format="rgb24")
                pending_indices.append(local_frame_index)
                pending_tensors.append(
                    preprocess_image(rgb, observation_size, model_size)
                )
                if len(pending_tensors) >= batch_size:
                    flush()

            if local_frame_index == episode.length - 1:
                flush()
                finish_episode()
                active = None
                cursor += 1
                if global_frame_index >= last_required_frame:
                    break

    if cursor != len(episodes):
        missing = [item.episode_index for item in episodes[cursor:]]
        raise RuntimeError(f"{job.video_path}: decode ended before episodes {missing[:8]}")


# Set once in main so process_video_job can keep the storage helper signature
# compatible with the pre-existing Bridge writer.
CONFIG_CAMERA_KEY = "observation.images.image"


def main() -> None:
    global CONFIG_CAMERA_KEY
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard-id < num-shards")

    config = load_config(args.config)
    if str(config["data"].get("dataset_format", "")) != "lerobot_v3":
        raise ValueError("This worker requires data.dataset_format=lerobot_v3")
    if int(config["data"]["horizon"]) != 1:
        raise ValueError("This LIBERO worker is intentionally restricted to data.horizon=1")
    CONFIG_CAMERA_KEY = str(config["data"]["camera"])
    output_root = resolve_config_path(config, config["paths"]["output_root"])
    checkpoint = resolve_config_path(config, config["paths"]["checkpoint"])
    dataset_root = resolve_config_path(config, config["paths"]["dataset_root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(output_root, args.shard_id)
    LOGGER.info(
        "config=%s shard=%d/%d device=%s log=%s",
        args.config,
        args.shard_id,
        args.num_shards,
        args.device,
        log_path,
    )

    all_jobs = discover_video_jobs(config)
    ensure_run_metadata(config, output_root, checkpoint, all_jobs)
    selected = filter_jobs(
        all_jobs,
        suite=args.suite,
        episode_index=args.episode_index,
    )
    assignments = assign_jobs_greedily(selected, args.num_shards)
    jobs = limit_job_episodes(assignments[args.shard_id], args.limit_episodes)
    assigned_episodes = [episode for job in jobs for episode in job.episodes]
    assigned_frames = sum(max(1, item.length - 1) for item in assigned_episodes)
    LOGGER.info(
        "discovered_jobs=%d selected_jobs=%d assigned_jobs=%d "
        "assigned_episodes=%d assigned_target_frames=%d",
        len(all_jobs),
        len(selected),
        len(jobs),
        len(assigned_episodes),
        assigned_frames,
    )
    for job in jobs:
        LOGGER.info(
            "job suite=%s chunk=%03d file=%03d episodes=%d targets=%d video=%s",
            job.suite_name,
            job.video_chunk_index,
            job.video_file_index,
            len(job.episodes),
            job.target_frames,
            job.video_path,
        )
    if args.dry_run:
        LOGGER.info("dry-run complete; no model loaded and no episode cache written")
        return

    horizon = int(config["data"]["horizon"])
    observation_size = tuple(map(int, config["preprocess"]["observation_size"]))
    model_size = tuple(map(int, config["preprocess"]["model_size"]))
    batch_size = int(config["teacher"]["batch_size_per_gpu"])
    episodes_per_directory = int(config["storage"]["episodes_per_directory"])
    resume = bool(config["runtime"]["resume"])
    fail_fast = bool(config["runtime"]["fail_fast"])
    manifest_path = output_root / "manifests" / f"shard_{args.shard_id:02d}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    counters = {"processed": 0, "skipped": 0, "failed": 0}
    manifest = manifest_path.open("a", encoding="utf-8", buffering=1)

    def record(payload: dict[str, Any]) -> None:
        status = str(payload["status"])
        counters[status] += 1
        manifest.write(json.dumps(payload, ensure_ascii=False) + "\n")
        completed = sum(counters.values())
        if completed == 1 or completed % int(config["runtime"]["log_every_episodes"]) == 0:
            LOGGER.info(
                "progress=%d/%d processed=%d skipped=%d failed=%d",
                completed,
                len(assigned_episodes),
                counters["processed"],
                counters["skipped"],
                counters["failed"],
            )

    pending_jobs: list[VideoJob] = []
    for job in jobs:
        pending: list[EpisodeDescriptor] = []
        for episode in job.episodes:
            destination = cache_path(
                output_root,
                episode.suite_name,
                episode.episode_index,
                episodes_per_directory,
            )
            if destination.is_file() and resume:
                stats = validate_episode_cache(
                    destination,
                    episode_index=episode.episode_index,
                    episode_length=episode.length,
                    model_size=model_size,
                    horizon=horizon,
                )
                record(
                    {
                        "status": "skipped",
                        "suite": episode.suite_name,
                        "cache_path": str(destination),
                        **stats,
                    }
                )
            else:
                pending.append(episode)
        if pending:
            pending_jobs.append(
                VideoJob(
                    suite_name=job.suite_name,
                    video_chunk_index=job.video_chunk_index,
                    video_file_index=job.video_file_index,
                    video_path=job.video_path,
                    episodes=tuple(pending),
                )
            )

    started = time.perf_counter()
    try:
        if pending_jobs:
            os.environ.setdefault("XFORMERS_DISABLED", "1")
            add_offline_sources(config)
            from moge.model.v3 import MoGeModel

            device = torch.device(args.device)
            random.seed(int(config["runtime"]["seed"]) + args.shard_id)
            torch.manual_seed(int(config["runtime"]["seed"]) + args.shard_id)
            load_started = time.perf_counter()
            model = MoGeModel.from_pretrained(str(checkpoint)).to(device).eval()
            model.requires_grad_(False)
            LOGGER.info("model_loaded_seconds=%.3f", time.perf_counter() - load_started)

            for job in pending_jobs:
                try:
                    process_video_job(
                        model,
                        job,
                        output_root=output_root,
                        horizon=horizon,
                        observation_size=observation_size,
                        model_size=model_size,
                        batch_size=batch_size,
                        device=device,
                        teacher_cfg=config["teacher"],
                        episodes_per_directory=episodes_per_directory,
                        record_callback=record,
                    )
                except Exception as error:
                    LOGGER.exception("video job failed: %s", job.video_path)
                    for episode in job.episodes:
                        destination = cache_path(
                            output_root,
                            episode.suite_name,
                            episode.episode_index,
                            episodes_per_directory,
                        )
                        if destination.is_file():
                            continue
                        record(
                            {
                                "status": "failed",
                                "suite": episode.suite_name,
                                "episode_index": episode.episode_index,
                                "episode_length": episode.length,
                                "source_video": str(job.video_path),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                    if fail_fast:
                        raise
    finally:
        manifest.close()

    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "assigned_jobs": len(jobs),
        "assigned_episodes": len(assigned_episodes),
        "assigned_target_frames": assigned_frames,
        **counters,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(
        output_root / "summaries" / f"shard_{args.shard_id:02d}.json",
        summary,
    )
    LOGGER.info("summary=%s", json.dumps(summary))
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
