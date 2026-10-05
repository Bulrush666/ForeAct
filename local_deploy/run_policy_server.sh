#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${COSMOS2_MODEL_DIR:-${REPO_DIR}/pretrained/cosmos2}"
CKPT_PATH="${COSMOS2_ACTION_CKPT:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10093}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "COSMOS2_ACTION_CKPT must point to a trained ForeAct checkpoint." >&2
  exit 2
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Action checkpoint not found: ${CKPT_PATH}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Cosmos-Predict2 directory not found: ${MODEL_DIR}" >&2
  exit 2
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --seed "${SEED}" \
  --action-seed "${SEED}" \
  --use_bf16 \
  --config_override "framework.world_model.base_wm=${MODEL_DIR}"
