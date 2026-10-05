#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/pretrained/cosmos2}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/bridge_orig_1.0.0_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/cosmos_depth_probe_episode_000000}"
GPU_ID="${GPU_ID:-7}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/eval_cosmos_depth_probe.py \
  --model "${MODEL_DIR}" \
  --dataset "${DATASET_DIR}" \
  --output "${OUTPUT_DIR}" \
  --device cuda \
  "$@"
