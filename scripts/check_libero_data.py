"""CPU-only, read-only LIBERO v3 metadata/depth preflight (no GPU allocation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yaml

SUITES = tuple(f"libero_{name}_no_noops_lerobot" for name in ("object", "goal", "spatial", "10"))
CAMERA = "observation.images.image"


def validate_contract(config):
    data = config["datasets"]["vla_data"]
    if data["data_mix"] != "libero_all_v30_main" or data["lerobot_version"] != "v3.0":
        raise ValueError("This launcher requires libero_all_v30_main / LeRobot v3.0")
    if data.get("include_state") or data.get("action_mode") != "abs":
        raise ValueError("LIBERO reference contract requires include_state=false, action_mode=abs")
    if config["framework"]["action_model"]["action_horizon"] != 8:
        raise ValueError("LIBERO DataConfig returns action[t:t+8]; action_horizon must be 8")
    if data.get("depth_cache_dataset_name"):
        raise ValueError("Do not share one depth_cache_dataset_name across the four suites")
    if data.get("depth_frame_offsets") != [1] or data.get("depth_cache_horizon") != 1:
        raise ValueError("LIBERO reference depth is depth[min(t+1, episode_length-1)]")
    if not data.get("require_depth") or not config["depth"].get("supervision_enabled"):
        raise ValueError("This config requires depth supervision; missing caches cannot silently disable it")


def check_depth(path, episode, length, size):
    """Check headers, identity and availability without loading large depth arrays."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing MoGe depth cache: {path}")
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        for key in ("depth", "valid_mask"):
            with archive.open(key + ".npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"Unsupported NPY header {version}: {path}")
                if dtype.hasobject:
                    raise ValueError(f"Object arrays are not allowed: {path}")
                shapes[key] = shape
    expected = (length, *size)
    if shapes["depth"] != expected or shapes["valid_mask"] != expected:
        raise ValueError(f"Depth length/size mismatch: {path}: {shapes}, expected {expected}")
    with np.load(path, allow_pickle=False) as cache:
        if int(cache["episode_index"].item()) != episode or int(cache["horizon"].item()) != 1:
            raise ValueError(f"Depth episode/horizon mismatch: {path}")
        available = cache["available"].astype(bool)
        if available.shape != (length,) or not available[min(1, length - 1):].all():
            raise ValueError(f"Depth t+1 target frames are incomplete: {path}")


def check_data(config, *, metadata_only=False):
    validate_contract(config)
    data = config["datasets"]["vla_data"]
    root = Path(data["data_root_dir"])
    depth_root = Path(data["depth_cache_root"])
    if not metadata_only and not depth_root.is_dir():
        raise FileNotFoundError(f"Depth root is missing: {depth_root}. Set DEPTH_CACHE_ROOT to the generated LIBERO t+1 cache root.")
    report = []
    for suite in SUITES:
        folder = root / suite
        info = json.loads((folder / "meta/info.json").read_text())
        modality = json.loads((folder / "meta/modality.json").read_text())
        if modality["video"]["primary_image"]["original_key"] != CAMERA:
            raise ValueError(f"Main camera mapping differs: {folder}")
        if info["codebase_version"] != "v3.0":
            raise ValueError(f"Not LeRobot v3.0: {folder}")
        for required in ("meta/tasks.parquet", "meta/stats_gr00t.json"):
            if not (folder / required).is_file():
                raise FileNotFoundError(folder / required)
        columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
        columns += [f"videos/{CAMERA}/{key}" for key in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
        files = sorted((folder / "meta/episodes").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No v3 episode metadata: {folder}")
        episodes = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
        if episodes.episode_index.duplicated().any() or len(episodes) != info["total_episodes"]:
            raise ValueError(f"Duplicate or incomplete episode metadata: {folder}")
        distinct_shards = 0
        for row in episodes.to_dict("records"):
            episode, length = int(row["episode_index"]), int(row["length"])
            dc, df = int(row["data/chunk_index"]), int(row["data/file_index"])
            vc, vf = int(row[f"videos/{CAMERA}/chunk_index"]), int(row[f"videos/{CAMERA}/file_index"])
            start, end = float(row[f"videos/{CAMERA}/from_timestamp"]), float(row[f"videos/{CAMERA}/to_timestamp"])
            if length <= 0 or not np.isfinite([start, end]).all() or start < 0 or end <= start:
                raise ValueError(f"Invalid episode/timestamp: {suite} episode {episode}")
            distinct_shards += (dc, df) != (vc, vf)
            for template, chunk, file in ((info["data_path"], dc, df), (info["video_path"], vc, vf)):
                path = folder / template.format(chunk_index=chunk, file_index=file, video_key=CAMERA, episode_index=episode)
                if not path.is_file():
                    raise FileNotFoundError(path)
            if not metadata_only:
                chunk = episode // int(data["depth_episodes_per_directory"])
                path = depth_root / "cache" / suite / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.npz"
                check_depth(path, episode, length, tuple(data["depth_model_size"]))
        item = dict(suite=suite, episodes=len(episodes), frames=int(episodes.length.sum()), video_data_shards_differ=distinct_shards)
        print(json.dumps(item), flush=True)
        report.append(item)
    print("Metadata OK; depth NOT checked (not training-ready)." if metadata_only else "Metadata and all required depth cache headers/availability OK; numerical depth quality is not assessed.", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/train/libero.yaml"))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    check_data(yaml.safe_load(Path(args.config).read_text()), metadata_only=args.metadata_only)
