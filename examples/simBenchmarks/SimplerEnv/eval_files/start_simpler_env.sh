#!/usr/bin/env bash
set -euo pipefail

FOREACT_DIR="${FOREACT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
SIM_PYTHON="${SIM_PYTHON:-python}"
SIMPLER_ENV_ROOT="${SIMPLER_ENV_ROOT:?Set SIMPLER_ENV_ROOT to the SimplerEnv checkout}"
SIMPLER_ENV_LIB_DIR="${SIMPLER_ENV_LIB_DIR:-}"
CKPT="${CKPT:?Set CKPT to a ForeAct checkpoint path}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6678}"
SEED="${SEED:-7}"
ACTION_SCALE="${ACTION_SCALE:-1.091}"
ENSEMBLE_HORIZON="${ENSEMBLE_HORIZON:-13}"
ENSEMBLE_ALPHA="${ENSEMBLE_ALPHA:-0.1}"
DENOISE_STEPS="${DENOISE_STEPS:-4}"
CONTROL_FREQ="${CONTROL_FREQ:-5}"
SIM_FREQ="${SIM_FREQ:-500}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-120}"
EPISODE_START="${EPISODE_START:-0}"
EPISODE_END="${EPISODE_END:-24}"
TASK="${TASK:-all}"

cd "${FOREACT_DIR}"
export PYTHONPATH="${FOREACT_DIR}:${PYTHONPATH:-}"
if [[ -n "${SIMPLER_ENV_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${SIMPLER_ENV_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

ckpt_dir="$(dirname "${CKPT}")"
ckpt_base="$(basename "${CKPT}")"
ckpt_name="${ckpt_base%.*}"
output_eval_dir="${ckpt_dir}/output_eval"
mkdir -p "${output_eval_dir}"

run_task() {
  local task_name="$1"
  if [[ "${TASK}" != "all" && "${TASK}" != "${task_name}" ]]; then
    return
  fi
  local env_name="$2"
  local scene_name="$3"
  local robot="$4"
  local overlay="$5"
  local robot_x="$6"
  local robot_y="$7"
  local task_log="${output_eval_dir}/${ckpt_name}_${task_name}_seed${SEED}.log"

  echo "Running ${task_name}: episodes ${EPISODE_START}-${EPISODE_END}"
  "${SIM_PYTHON}" examples/simBenchmarks/SimplerEnv/eval_files/start_simpler_env.py \
    --ckpt-path "${CKPT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --robot "${robot}" \
    --policy-setup widowx_bridge \
    --action-scale "${ACTION_SCALE}" \
    --action-ensemble-horizon "${ENSEMBLE_HORIZON}" \
    --adaptive-ensemble-alpha "${ENSEMBLE_ALPHA}" \
    --num-ddim-steps "${DENOISE_STEPS}" \
    --control-freq "${CONTROL_FREQ}" \
    --sim-freq "${SIM_FREQ}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --octo-init-rng "${SEED}" \
    --env-name "${env_name}" \
    --scene-name "${scene_name}" \
    --rgb-overlay-path "${overlay}" \
    --robot-init-x-range "${robot_x}" "${robot_x}" 1 \
    --robot-init-y-range "${robot_y}" "${robot_y}" 1 \
    --obj-variation-mode episode \
    --obj-episode-range "${EPISODE_START}" "${EPISODE_END}" \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    2>&1 | tee "${task_log}"
}

bridge_overlay="${SIMPLER_ENV_ROOT}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
sink_overlay="${SIMPLER_ENV_ROOT}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"

run_task spoon PutSpoonOnTableClothInScene-v0 bridge_table_1_v1 widowx "${bridge_overlay}" 0.147 0.028
run_task carrot PutCarrotOnPlateInScene-v0 bridge_table_1_v1 widowx "${bridge_overlay}" 0.147 0.028
run_task cube StackGreenCubeOnYellowCubeBakedTexInScene-v0 bridge_table_1_v1 widowx "${bridge_overlay}" 0.147 0.028
run_task eggplant PutEggplantInBasketScene-v0 bridge_table_1_v2 widowx_sink_camera_setup "${sink_overlay}" 0.127 0.060
