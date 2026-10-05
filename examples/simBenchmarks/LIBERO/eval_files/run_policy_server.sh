#!/usr/bin/env bash
set -euo pipefail

FOREACT_DIR="${FOREACT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
FOREACT_PYTHON="${FOREACT_PYTHON:-python}"
CKPT="${CKPT:?Set CKPT to a ForeAct checkpoint path}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
DENOISE_STEPS="${DENOISE_STEPS:-10}"
USE_BF16="${USE_BF16:-1}"

cd "${FOREACT_DIR}"
export PYTHONPATH="${FOREACT_DIR}:${PYTHONPATH:-}"

cmd=(
  "${FOREACT_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --host "${HOST}"
  --port "${PORT}"
  --seed "${SEED}"
  --action-seed "${SEED}"
  --config_override "framework.action_model.num_inference_timesteps=${DENOISE_STEPS}"
  --config_override "depth.enabled=false"
)

if [[ "${USE_BF16}" == "1" ]]; then
  cmd+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
