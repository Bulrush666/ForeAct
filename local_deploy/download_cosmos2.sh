#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${COSMOS2_MODEL_DIR:-${REPO_DIR}/pretrained/cosmos2}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_BIN="${HF_BIN:-hf}"
REVISION="f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"

mkdir -p "${MODEL_DIR}"
echo "Downloading nvidia/Cosmos-Predict2-2B-Video2World to ${MODEL_DIR} via ${HF_ENDPOINT}"
echo "The download is resumable; rerun this script after a transient network failure."

"${HF_BIN}" download \
  nvidia/Cosmos-Predict2-2B-Video2World \
  --revision "${REVISION}" \
  --include \
    'config.json' \
    'model_index.json' \
    'scheduler/*' \
    'text_encoder/*' \
    'tokenizer/*' \
    'transformer/*' \
    'vae/*' \
  --local-dir "${MODEL_DIR}"

required_files=(
  "scheduler/scheduler_config.json"
  "text_encoder/config.json"
  "text_encoder/model.safetensors.index.json"
  "tokenizer/tokenizer_config.json"
  "transformer/config.json"
  "transformer/diffusion_pytorch_model.safetensors"
  "vae/config.json"
  "vae/diffusion_pytorch_model.safetensors"
)
for relpath in "${required_files[@]}"; do
  if [[ ! -s "${MODEL_DIR}/${relpath}" ]]; then
    echo "Missing required file after download: ${MODEL_DIR}/${relpath}" >&2
    exit 1
  fi
done

echo "Model files:" 
find "${MODEL_DIR}" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort
