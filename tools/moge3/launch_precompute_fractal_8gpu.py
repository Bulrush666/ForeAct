#!/usr/bin/env python3
"""Launch eight independent Fractal next-frame MoGe-3 cache workers."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from moge3_depth.config import load_config, resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "fractal_next1.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-episodes-per-worker", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    # Preserve the venv interpreter symlink. Path.resolve() would collapse
    # moge3_env/bin/python back to its foreact6 base interpreter and skip the
    # venv's Triton/utils3d overrides.
    python_bin = Path(config["paths"]["python_bin"]).expanduser()
    if not python_bin.is_absolute():
        python_bin = Path(config["_config_path"]).parent / python_bin
    python_bin = python_bin.absolute()
    output_root = resolve_config_path(config, config["paths"]["output_root"])
    moge_source = resolve_config_path(config, config["paths"]["moge_source_dir"])
    flex_source = resolve_config_path(config, config["paths"]["flex_gemm_source_dir"])
    gpu_ids = [str(value) for value in config["distributed"]["gpu_ids"]]
    expected_workers = int(config["distributed"]["num_workers"])
    if len(gpu_ids) != expected_workers:
        raise ValueError(
            f"distributed.num_workers={expected_workers}, but gpu_ids={gpu_ids}"
        )
    if expected_workers != 8:
        raise ValueError("This launch contract expects exactly eight workers/GPUs")
    for path in (python_bin, moge_source, flex_source):
        if not path.exists():
            raise FileNotFoundError(path)

    output_root.mkdir(parents=True, exist_ok=True)
    launcher_logs = output_root / "logs" / "launcher"
    launcher_logs.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "precompute_fractal_depth.py"
    processes: list[tuple[subprocess.Popen, object, list[str], str]] = []
    stagger = float(config["distributed"].get("stagger_seconds", 15))

    try:
        for shard_id, gpu_id in enumerate(gpu_ids):
            command = [
                str(python_bin),
                "-u",
                str(script),
                "--config",
                str(Path(args.config).resolve()),
                "--shard-id",
                str(shard_id),
                "--num-shards",
                str(expected_workers),
                "--device",
                "cuda:0",
            ]
            if args.limit_episodes_per_worker is not None:
                command.extend(
                    ["--limit-episodes", str(args.limit_episodes_per_worker)]
                )
            if args.dry_run:
                command.append("--dry-run")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["PYTHONNOUSERSITE"] = "1"
            env["XFORMERS_DISABLED"] = "1"
            existing_pythonpath = env.get("PYTHONPATH", "")
            entries = [str(script.parent), str(moge_source), str(flex_source)]
            if existing_pythonpath:
                entries.append(existing_pythonpath)
            env["PYTHONPATH"] = os.pathsep.join(entries)
            log_path = launcher_logs / f"gpu_{gpu_id}_shard_{shard_id:02d}.log"
            print(f"launch shard={shard_id} physical_gpu={gpu_id} log={log_path}")
            print(" ".join(command))
            if args.dry_run:
                continue
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, log_handle, command, gpu_id))
            if shard_id + 1 < len(gpu_ids):
                time.sleep(stagger)

        if args.dry_run:
            return
        failures = []
        for process, log_handle, command, gpu_id in processes:
            return_code = process.wait()
            log_handle.close()
            print(f"worker physical_gpu={gpu_id} return_code={return_code}")
            if return_code:
                failures.append((gpu_id, return_code, command))
        if failures:
            for gpu_id, return_code, command in failures:
                print(
                    f"FAILED gpu={gpu_id} return_code={return_code}: {' '.join(command)}",
                    file=sys.stderr,
                )
            raise SystemExit(1)
    except KeyboardInterrupt:
        print("interrupt received; terminating all workers", file=sys.stderr)
        for process, _, _, _ in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise
    finally:
        for process, log_handle, _, _ in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            if not log_handle.closed:
                log_handle.close()


if __name__ == "__main__":
    main()
