#!/usr/bin/env python3
"""Validate LIBERO t+1 MoGe episode caches against LeRobot v3 metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moge3_depth.config import load_config, resolve_config_path, target_frame_indices
from moge3_depth.storage import cache_path, validate_episode_cache
from precompute_libero_depth import discover_video_jobs, filter_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "libero_next1.yaml")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = resolve_config_path(config, config["paths"]["output_root"])
    horizon = int(config["data"]["horizon"])
    model_size = tuple(map(int, config["preprocess"]["model_size"]))
    episodes_per_directory = int(config["storage"]["episodes_per_directory"])
    camera = str(config["data"]["camera"])

    jobs = filter_jobs(
        discover_video_jobs(config),
        suite=args.suite,
        episode_index=args.episode_index,
    )
    episodes = [episode for job in jobs for episode in job.episodes]
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]

    summary = {
        "expected": len(episodes),
        "valid": 0,
        "missing": 0,
        "invalid": 0,
        "available_frames": 0,
        "size_bytes": 0,
    }
    for episode in episodes:
        path = cache_path(
            output_root,
            episode.suite_name,
            episode.episode_index,
            episodes_per_directory,
        )
        if not path.is_file():
            summary["missing"] += 1
            print(f"MISSING {episode.suite_name} episode={episode.episode_index} path={path}")
            continue
        try:
            stats = validate_episode_cache(
                path,
                episode_index=episode.episode_index,
                episode_length=episode.length,
                model_size=model_size,
                horizon=horizon,
            )
            with np.load(path, allow_pickle=False) as archive:
                expected = np.zeros(episode.length, dtype=np.bool_)
                expected[target_frame_indices(episode.length, horizon)] = True
                actual = np.asarray(archive["available"], dtype=np.bool_)
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"available mask mismatch: expected {int(expected.sum())}, "
                        f"got {int(actual.sum())}"
                    )
                stored_camera = str(np.asarray(archive["camera"]).item())
                if stored_camera != camera:
                    raise ValueError(
                        f"camera mismatch: expected {camera!r}, got {stored_camera!r}"
                    )
            summary["valid"] += 1
            summary["available_frames"] += int(stats["available_frames"])
            summary["size_bytes"] += int(stats["size_bytes"])
        except Exception as error:
            summary["invalid"] += 1
            print(
                f"INVALID {episode.suite_name} episode={episode.episode_index} "
                f"path={path}: {type(error).__name__}: {error}"
            )

    print(json.dumps(summary, indent=2))
    if (summary["missing"] or summary["invalid"]) and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

