"""Read-only checks for the local Cosmos-Predict2 foreact deployment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("COSMOS2_MODEL_DIR", REPO / "pretrained/cosmos2"))
DATASET = REPO / "local_deploy/data/bridge_orig_1.0.0_lerobot"


def check(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    print(f"[OK] {label}: {path}")


def main() -> int:
    print(f"repo: {REPO}")
    print(f"python: {sys.executable}")
    check(REPO / "foreact/model/framework/WM4A/CosmoPredict2GR00T.py", "Cosmos framework")
    check(REPO / "configs/train/bridge.yaml", "top-level training config")
    check(DATASET / "data", "dataset data")
    check(DATASET / "videos", "dataset videos")
    check(DATASET / "meta/modality.json", "dataset modality")
    check(DATASET / "meta/info.json", "dataset info")

    modality = json.loads((DATASET / "meta/modality.json").read_text())
    required = {"state", "action", "video", "annotation"}
    missing = required.difference(modality)
    if missing:
        raise RuntimeError(f"modality.json missing sections: {sorted(missing)}")
    print(f"[OK] modality sections: {sorted(modality)}")

    required_model_files = [
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/spiece.model",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    missing_model_files = [relpath for relpath in required_model_files if not (MODEL / relpath).is_file()]
    if not missing_model_files:
        print(f"[OK] complete Cosmos Diffusers model: {len(required_model_files)} required files")
    else:
        print(f"[WAIT] Cosmos weights are not present yet: {MODEL}")
        print(f"      Missing {len(missing_model_files)} required files; first missing: {missing_model_files[0]}")
        print("      Run local_deploy/download_cosmos2.sh when Hugging Face is reachable.")

    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        import torch
        from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler  # noqa: F401
        from transformers import T5EncoderModel, T5TokenizerFast  # noqa: F401
        print(f"[OK] torch={torch.__version__}, diffusers={diffusers.__version__}")
    except Exception as exc:
        print(f"[FAIL] model dependencies: {exc}")
        return 1

    sys.path.insert(0, str(REPO))
    try:
        from foreact.model.modules.world_model.CosmoPredict2 import _CosmoPredict2_Interface  # noqa: F401
        print("[OK] foreact Cosmos wrapper import")
    except Exception as exc:
        print(f"[FAIL] foreact Cosmos wrapper import: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
