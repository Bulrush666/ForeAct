#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" local_deploy/verify_deployment.py
"${PYTHON_BIN}" foreact/dataloader/lerobot_datasets.py \
  --config_yaml train.yaml \
  --data_root_dir "${REPO_DIR}/local_deploy/data"
