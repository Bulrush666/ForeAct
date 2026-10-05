#!/usr/bin/env python3
"""Validate cache schema, padded endpoint indexing, and depth statistics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from moge3_depth.config import load_config, resolve_config_path, target_frame_index
from moge3_depth.storage import cache_path, load_episodes, validate_episode_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "train.yaml")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = resolve_config_path(config, config["paths"]["dataset_root"])
    output_root = resolve_config_path(config, config["paths"]["output_root"])
    episodes = load_episodes(dataset / "meta" / "episodes.jsonl")
    dataset_name = config["data"].get("dataset_name", dataset.name)
    horizon = int(config["data"]["horizon"])
    model_size = tuple(map(int, config["preprocess"]["model_size"]))
    episodes_per_directory = int(config["storage"]["episodes_per_directory"])

    existing = []
    missing = []
    for episode in episodes:
        path = cache_path(
            output_root,
            dataset_name,
            int(episode["episode_index"]),
            episodes_per_directory,
        )
        (existing if path.is_file() else missing).append((episode, path))
    print(
        json.dumps(
            {
                "total_episodes": len(episodes),
                "cached_episodes": len(existing),
                "missing_episodes": len(missing),
            },
            indent=2,
        )
    )
    if args.require_complete and missing:
        raise SystemExit(f"Cache is incomplete; first missing file: {missing[0][1]}")
    if not existing:
        raise SystemExit("No episode caches found")

    selected = random.Random(args.seed).sample(existing, min(args.count, len(existing)))
    records = []
    for episode, path in selected:
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        record = validate_episode_cache(
            path,
            episode_index=episode_index,
            episode_length=episode_length,
            model_size=model_size,
            horizon=horizon,
        )
        with np.load(path, allow_pickle=False) as archive:
            available = archive["available"]
            probes = sorted({0, episode_length // 2, episode_length - 1})
            for base_index in probes:
                target = target_frame_index(base_index, episode_length, horizon)
                if not bool(available[target]):
                    raise ValueError(
                        f"{path}: target frame {target} for base {base_index} unavailable"
                    )
        records.append(record)

    report = {
        "checked_episodes": len(records),
        "mean_available_frames": float(
            np.mean([record["available_frames"] for record in records])
        ),
        "mean_valid_ratio": float(
            np.mean([record["valid_ratio_available"] for record in records])
        ),
        "depth_median_of_episode_medians_m": float(
            np.median([record["depth_median_m"] for record in records])
        ),
        "sample_cache_size_mean_bytes": float(
            np.mean([record["size_bytes"] for record in records])
        ),
        "sample_cache_size_total_bytes": int(
            sum(record["size_bytes"] for record in records)
        ),
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
