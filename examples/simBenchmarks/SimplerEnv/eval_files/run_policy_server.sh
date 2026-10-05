#!/usr/bin/env bash
set -euo pipefail

FOREACT_DIR="${FOREACT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
FOREACT_PYTHON="${FOREACT_PYTHON:-python}"
CKPT="${CKPT:?Set CKPT to a ForeAct checkpoint path}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6678}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-7}"
DENOISE_STEPS="${DENOISE_STEPS:-4}"
USE_BF16="${USE_BF16:-1}"

cd "${FOREACT_DIR}"
export PYTHONPATH="${FOREACT_DIR}:${PYTHONPATH:-}"

ckpt_dir="$(dirname "${CKPT}")"
ckpt_base="$(basename "${CKPT}")"
ckpt_name="${ckpt_base%.*}"
log_dir="${ckpt_dir}/output_server"
mkdir -p "${log_dir}"
log_file="${log_dir}/${ckpt_name}_policy_server_${PORT}.log"

cmd=(
  "${FOREACT_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --host "${HOST}"
  --port "${PORT}"
  --seed "${SEED}"
  --action-seed "${SEED}"
  --config_override "framework.action_model.num_inference_timesteps=${DENOISE_STEPS}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  cmd+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" 2>&1 | tee "${log_file}"
