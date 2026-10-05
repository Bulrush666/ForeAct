#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/train/bridge.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Training config not found: ${CONFIG}" >&2
  exit 1
fi

# Parse the runtime section and pass the resolved values to the ForeAct trainer.
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python3}"
if ! command -v "${BOOTSTRAP_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${BOOTSTRAP_PYTHON}" ]]; then
  echo "Python is not executable: ${BOOTSTRAP_PYTHON}" >&2
  exit 1
fi

RUNTIME_FILE="$(mktemp)"
trap 'rm -f -- "${RUNTIME_FILE}"' EXIT
"${BOOTSTRAP_PYTHON}" - "${CONFIG}" "${PROJECT_ROOT}" >"${RUNTIME_FILE}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
project_root = Path(sys.argv[2]).expanduser().resolve()
raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
runtime = raw.get("runtime") or {}

defaults = {
    "python_bin": "python3",
    "accelerate_config": "foreact/config/deepspeed/zero2.yaml",
    "model_dir": "pretrained/cosmos2",
    "data_root_dir": "local_deploy/data",
    "output_root": "log",
    "run_name": "bridge_orig_cosmos_predict2_2b_token",
    "append_timestamp": True,
    "gpu_ids": "0",
    "nproc_per_node": None,
    "main_process_port": 29521,
    "steps": 100000,
    "per_device_batch_size": 8,
    "num_workers": 1,
    "save_every": 5000,
    "log_every": 10,
    "warmup_steps": 5000,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "resume": "none",
    "write_checkpoints": True,
    "wandb_mode": "offline",
    "omp_num_threads": 8,
    "cuda_allocator_config": "expandable_segments:True",
    "nccl_async_error_handling": 1,
}
values = {**defaults, **runtime}

def resolve(value: object) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())

def executable(value: object) -> str:
    command = str(value)
    if "/" not in command and "\\" not in command:
        return command
    return resolve(command)

def boolean(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    value = str(value).lower().strip()
    if value in {"1", "true", "yes", "on"}:
        return "1"
    if value in {"0", "false", "no", "off"}:
        return "0"
    raise ValueError(f"Invalid boolean value: {value!r}")

ordered = [
    executable(values["python_bin"]),
    resolve(values["accelerate_config"]),
    resolve(values["model_dir"]),
    resolve(values["data_root_dir"]),
    resolve(values["output_root"]),
    str(values["run_name"]),
    boolean(values["append_timestamp"]),
    str(values["gpu_ids"]),
    "" if values["nproc_per_node"] is None else str(values["nproc_per_node"]),
    str(values["main_process_port"]),
    str(values["steps"]),
    str(values["per_device_batch_size"]),
    str(values["num_workers"]),
    str(values["save_every"]),
    str(values["log_every"]),
    str(values["warmup_steps"]),
    str(values["seed"]),
    str(values["gradient_accumulation_steps"]),
    str(values["resume"]),
    boolean(values["write_checkpoints"]),
    str(values["wandb_mode"]),
    str(values["omp_num_threads"]),
    str(values["cuda_allocator_config"]),
    str(values["nccl_async_error_handling"]),
]
print("\n".join(ordered))
PY

mapfile -t SETTINGS <"${RUNTIME_FILE}"
if [[ "${#SETTINGS[@]}" -ne 24 ]]; then
  echo "Failed to parse runtime settings from ${CONFIG}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${SETTINGS[0]}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${SETTINGS[1]}}"
MODEL_DIR="${MODEL_DIR:-${SETTINGS[2]}}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-${SETTINGS[3]}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SETTINGS[4]}}"
RUN_NAME_BASE="${RUN_NAME_BASE:-${SETTINGS[5]}}"
APPEND_TIMESTAMP="${APPEND_TIMESTAMP:-${SETTINGS[6]}}"
GPU_IDS="${GPU_IDS:-${SETTINGS[7]}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SETTINGS[8]}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${SETTINGS[9]}}"
MAX_STEPS="${MAX_STEPS:-${SETTINGS[10]}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-${SETTINGS[11]}}"
NUM_WORKERS="${NUM_WORKERS:-${SETTINGS[12]}}"
SAVE_EVERY="${SAVE_EVERY:-${SETTINGS[13]}}"
LOG_EVERY="${LOG_EVERY:-${SETTINGS[14]}}"
WARMUP_STEPS="${WARMUP_STEPS:-${SETTINGS[15]}}"
SEED="${SEED:-${SETTINGS[16]}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${SETTINGS[17]}}"
RESUME="${RESUME:-${SETTINGS[18]}}"
WRITE_CHECKPOINTS="${WRITE_CHECKPOINTS:-${SETTINGS[19]}}"
WANDB_MODE="${WANDB_MODE:-${SETTINGS[20]}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SETTINGS[21]}}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${SETTINGS[22]}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ -z "${NPROC_PER_NODE}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
  NPROC_PER_NODE="${#GPU_ARRAY[@]}"
fi
if [[ "${GRADIENT_ACCUMULATION_STEPS}" != "1" ]]; then
  echo "This launcher uses the official DeepSpeed config with accumulation=1; set runtime.gradient_accumulation_steps to 1." >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT_DIR}/bridge_orig_1.0.0_lerobot" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "Bridge dataset overlay not found under: ${DATA_ROOT_DIR}" >&2
  exit 1
fi
required_model_files=(
  "scheduler/scheduler_config.json"
  "text_encoder/config.json"
  "text_encoder/model-00001-of-00002.safetensors"
  "text_encoder/model-00002-of-00002.safetensors"
  "text_encoder/model.safetensors.index.json"
  "tokenizer/tokenizer_config.json"
  "tokenizer/tokenizer.json"
  "tokenizer/spiece.model"
  "transformer/config.json"
  "transformer/diffusion_pytorch_model.safetensors"
  "vae/config.json"
  "vae/diffusion_pytorch_model.safetensors"
)
missing_model_file=""
for relpath in "${required_model_files[@]}"; do
  if [[ ! -s "${MODEL_DIR}/${relpath}" ]]; then
    missing_model_file="${relpath}"
    break
  fi
done
if [[ -n "${missing_model_file}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "Cosmos weights are incomplete or not downloaded: ${MODEL_DIR}" >&2
  echo "First missing file: ${missing_model_file}" >&2
  echo "Run local_deploy/download_cosmos2.sh after Hugging Face authorization." >&2
  exit 1
fi

if [[ -n "${RUN_NAME:-}" ]]; then
  RESOLVED_RUN_NAME="${RUN_NAME}"
elif [[ "${APPEND_TIMESTAMP}" == "1" ]]; then
  RESOLVED_RUN_NAME="${RUN_NAME_BASE}_$(date +%Y%m%d_%H%M%S)"
else
  RESOLVED_RUN_NAME="${RUN_NAME_BASE}"
fi
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RESOLVED_RUN_NAME}}"
mkdir -p "${RUN_DIR}"
if ! touch "${RUN_DIR}/.write_test" 2>/dev/null; then
  echo "Run directory is not writable: ${RUN_DIR}" >&2
  exit 1
fi
rm -f "${RUN_DIR}/.write_test"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF
export WANDB_MODE

TRAIN_ARGS=(
  --config_yaml "${CONFIG}"
  --framework.world_model.base_wm "${MODEL_DIR}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --datasets.vla_data.num_workers "${NUM_WORKERS}"
  --trainer.max_train_steps "${MAX_STEPS}"
  --trainer.save_interval "${SAVE_EVERY}"
  --trainer.logging_frequency "${LOG_EVERY}"
  --trainer.num_warmup_steps "${WARMUP_STEPS}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --run_root_dir "${OUTPUT_ROOT}"
  --run_id "${RESOLVED_RUN_NAME}"
)

if [[ "${RESUME}" == "auto" ]]; then
  TRAIN_ARGS+=(--trainer.is_resume true)
elif [[ -n "${RESUME}" && "${RESUME}" != "none" && "${RESUME}" != "false" ]]; then
  TRAIN_ARGS+=(--trainer.pretrained_checkpoint "${RESUME}")
fi
if [[ "${WRITE_CHECKPOINTS}" != "1" ]]; then
  echo "Warning: official trainer always writes final_model; periodic checkpoint suppression is not supported." >&2
fi

cd "${PROJECT_ROOT}"
echo "ForeAct Bridge training"
echo "  config=${CONFIG}"
echo "  run_dir=${RUN_DIR}"
echo "  model_dir=${MODEL_DIR}"
echo "  data_root_dir=${DATA_ROOT_DIR}"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "  nproc_per_node=${NPROC_PER_NODE}"
echo "  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
echo "  global_batch_size=$((PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE))"
echo "  wandb_mode=${WANDB_MODE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'accelerate command:'
  printf ' %q' "${PYTHON_BIN}" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NPROC_PER_NODE}" --main_process_port "${MAIN_PROCESS_PORT}" foreact/training/train.py "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec "${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NPROC_PER_NODE}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  foreact/training/train.py \
  "${TRAIN_ARGS[@]}"
