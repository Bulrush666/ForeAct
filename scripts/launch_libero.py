"""Launch the existing ForeAct trainer with an isolated, checked LIBERO config."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

from check_libero_data import check_data, validate_contract

ROOT = Path(__file__).resolve().parents[1]


def resolve_config(config, environ):
    config = copy.deepcopy(config)
    runtime = config["runtime"]
    overrides = {
        "GPU_IDS": ("gpu_ids", str), "NPROC_PER_NODE": ("nproc_per_node", int),
        "MODEL_DIR": ("model_dir", str), "DATA_ROOT_DIR": ("data_root_dir", str),
        "OUTPUT_ROOT": ("output_root", str), "PER_DEVICE_BATCH_SIZE": ("per_device_batch_size", int),
        "NUM_WORKERS": ("num_workers", int), "MAX_STEPS": ("steps", int),
        "MAIN_PROCESS_PORT": ("main_process_port", int), "SEED": ("seed", int),
        "PYTHON_BIN": ("python_bin", str), "WANDB_MODE": ("wandb_mode", str),
    }
    for env_key, (key, cast) in overrides.items():
        if env_key in environ:
            runtime[key] = cast(environ[env_key])
    # Paths are relative to the repository, never the caller's current directory.
    for key in ("model_dir", "data_root_dir", "output_root", "accelerate_config"):
        runtime[key] = str((ROOT / Path(runtime[key]).expanduser()).resolve())
    python_bin = str(runtime["python_bin"])
    runtime["python_bin"] = (
        shutil.which(python_bin)
        if "/" not in python_bin and "\\" not in python_bin
        else str((ROOT / Path(python_bin).expanduser()).resolve())
    )
    if not runtime["python_bin"]:
        raise FileNotFoundError(f"Python executable not found: {python_bin}")
    data, trainer = config["datasets"]["vla_data"], config["trainer"]
    data.update(data_root_dir=runtime["data_root_dir"], per_device_batch_size=runtime["per_device_batch_size"], num_workers=runtime["num_workers"])
    data["depth_cache_root"] = str((ROOT / Path(environ.get("DEPTH_CACHE_ROOT", data["depth_cache_root"])).expanduser()).resolve())
    config["framework"]["world_model"]["base_wm"] = runtime["model_dir"]
    for key, runtime_key in (("max_train_steps", "steps"), ("num_warmup_steps", "warmup_steps"), ("save_interval", "save_every"), ("logging_frequency", "log_every"), ("gradient_accumulation_steps", "gradient_accumulation_steps")):
        trainer[key] = runtime[runtime_key]
    config["seed"] = runtime["seed"]
    config["run_root_dir"] = runtime["output_root"]
    config["run_id"] = environ.get("RUN_NAME", runtime["run_name"] + (datetime.now().strftime("_%Y%m%d_%H%M%S") if runtime["append_timestamp"] else ""))
    gpu_ids = [value.strip() for value in runtime["gpu_ids"].split(",") if value.strip()]
    nproc = runtime.get("nproc_per_node") or len(gpu_ids)
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or nproc != len(gpu_ids):
        raise ValueError("NPROC_PER_NODE must equal the number of distinct GPU_IDS")
    runtime["nproc_per_node"] = nproc
    if runtime["gradient_accumulation_steps"] != 1:
        raise ValueError("This reference launcher supports accumulation=1 only")
    if runtime["per_device_batch_size"] < 1 or runtime["num_workers"] < 0:
        raise ValueError("Invalid batch size or worker count")
    if runtime["resume"] != "none" or not runtime["write_checkpoints"]:
        raise ValueError("This launcher creates fresh runs; resume must be none and write_checkpoints=true")
    validate_contract(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/train/libero.yaml")
    )
    parser.add_argument("--dry-run", action="store_true", help="Check RGB metadata and print command; no files written and no training")
    args = parser.parse_args()
    config = resolve_config(yaml.safe_load(Path(args.config).read_text()), os.environ)
    runtime = config["runtime"]
    dry_run = args.dry_run or os.environ.get("DRY_RUN") == "1"
    check_data(config, metadata_only=dry_run)
    for key in ("python_bin", "accelerate_config"):
        if not Path(runtime[key]).is_file():
            raise FileNotFoundError(runtime[key])
    model = Path(runtime["model_dir"])
    required = ["scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json", "tokenizer/spiece.model", "text_encoder/config.json", "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
    for relative in required:
        if not (model / relative).is_file():
            raise FileNotFoundError(model / relative)
    index = model / "text_encoder/model.safetensors.index.json"
    if index.is_file():
        import json
        for filename in set(json.loads(index.read_text())["weight_map"].values()):
            if not (index.parent / filename).is_file():
                raise FileNotFoundError(index.parent / filename)
    elif not (model / "text_encoder/model.safetensors").is_file():
        raise FileNotFoundError("Missing local T5 safetensors/index")
    run_dir = Path(config["run_root_dir"]) / config["run_id"]
    resolved = run_dir / "launch_config.yaml"
    command = [runtime["python_bin"], "-m", "accelerate.commands.launch", "--config_file", runtime["accelerate_config"], "--num_processes", str(runtime["nproc_per_node"]), "--main_process_port", str(runtime["main_process_port"]), str(ROOT / "foreact/training/train.py"), "--config_yaml", str(resolved)]
    print(f"GPUs={runtime['gpu_ids']}; batch/GPU={runtime['per_device_batch_size']}; global batch={runtime['per_device_batch_size'] * runtime['nproc_per_node']}", flush=True)
    print(shlex.join(command), flush=True)
    if dry_run:
        print("DRY RUN: no training, no output directory created; depth cache must pass checks before a real launch.")
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=runtime["gpu_ids"], PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""), TOKENIZERS_PARALLELISM="false", NO_ALBUMENTATIONS_UPDATE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE=runtime["wandb_mode"], OMP_NUM_THREADS=str(runtime["omp_num_threads"]), PYTORCH_CUDA_ALLOC_CONF=runtime["cuda_allocator_config"], TORCH_NCCL_ASYNC_ERROR_HANDLING=str(runtime["nccl_async_error_handling"]))
    with (run_dir / "console.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    raise SystemExit(status)


if __name__ == "__main__":
    main()
