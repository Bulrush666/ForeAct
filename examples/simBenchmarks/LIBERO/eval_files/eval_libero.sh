#!/usr/bin/env bash
set -euo pipefail

FOREACT_DIR="${FOREACT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:?Set LIBERO_HOME to the LIBERO checkout}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CKPT="${CKPT:?Set CKPT to a ForeAct checkpoint path}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-42}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-osmesa}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-osmesa}"

cd "${FOREACT_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${FOREACT_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

ckpt_dir="$(dirname "${CKPT}")"
ckpt_base="$(basename "${CKPT}")"
ckpt_name="${ckpt_base%.*}"
video_out_path="${ckpt_dir}/results/${TASK_SUITE_NAME}/${ckpt_name}"

"${LIBERO_PYTHON}" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.seed "${SEED}" \
  --args.unnorm-key franka \
  --args.video-out-path "${video_out_path}"
