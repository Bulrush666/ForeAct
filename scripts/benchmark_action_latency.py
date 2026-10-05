#!/usr/bin/env python3
"""Benchmark the action-only latency of a Cosmos2 WorldVLA checkpoint.

The timed boundary is one call to ``model.predict_action`` with an image and an
instruction already resident in host memory.  That call performs:

    image -> Cosmos VAE encode
    instruction -> T5 encode
    one Cosmos transformer forward
    ActionDiT Euler integration (normally four steps)
    normalized action chunk copied back to host memory

Checkpoint loading and image-file decoding are intentionally excluded.  Depth
decoding is forcibly disabled so this measures the policy path used to produce
the action chunk, not the auxiliary training head.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from foreact.model.framework.base_framework import baseframework  # noqa: E402


ACTION_ONLY_OVERRIDES = (
    "depth.enabled=false",
    "depth.supervision_enabled=false",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure image + T5 + Cosmos single-forward + ActionDiT latency. "
            "Model loading, image-file I/O, and depth decoding are excluded."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Current RGB frame. If omitted, use a deterministic 224x224 frame.",
    )
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model parameter dtype. bf16 matches the normal evaluation setup.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Untimed warm-up calls before steady-state measurement.",
    )
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument(
        "--expected-action-horizon",
        type=int,
        default=16,
        help="Fail if the checkpoint does not output this many actions.",
    )
    parser.add_argument(
        "--expected-action-steps",
        type=int,
        default=4,
        help="Fail if ActionDiT is not configured for this many Euler steps.",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Additional checkpoint config override; may be repeated. "
            "Action-only depth overrides are always applied last."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the machine-readable result.",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.runs <= 0:
        parser.error("--runs must be > 0")
    if not args.device.startswith("cuda"):
        parser.error("This Cosmos2 implementation requires a CUDA device.")
    return args


def load_current_frame(path: Path | None) -> Image.Image:
    if path is not None:
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    # Deterministic, non-uniform data avoids special-casing an all-zero image.
    height, width = 224, 224
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            xx * 255 // max(width - 1, 1),
            yy * 255 // max(height - 1, 1),
            (xx + yy) * 255 // max(width + height - 2, 1),
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def timed_predict(model: torch.nn.Module, example: dict[str, Any], device: torch.device):
    synchronize(device)
    started = time.perf_counter()
    output = model.predict_action(examples=[example])
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, output


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    samples = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(samples.mean()),
        "std_ms": float(samples.std(ddof=0)),
        "min_ms": float(samples.min()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p90_ms": float(np.percentile(samples, 90)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "max_ms": float(samples.max()),
    }


def parameter_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def validate_model_contract(
    model: torch.nn.Module,
    expected_horizon: int,
    expected_steps: int,
) -> tuple[int, int]:
    horizon = int(getattr(model, "action_horizon"))
    steps = int(getattr(model.action_model, "num_inference_timesteps"))
    if horizon != expected_horizon:
        raise RuntimeError(
            f"Checkpoint action_horizon={horizon}, expected {expected_horizon}. "
            "Pass --expected-action-horizon to benchmark this checkpoint intentionally."
        )
    if steps != expected_steps:
        raise RuntimeError(
            f"Checkpoint ActionDiT num_inference_timesteps={steps}, expected {expected_steps}. "
            "Pass --expected-action-steps to benchmark this setting intentionally."
        )
    return horizon, steps


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Apply action-only settings last so a benchmark cannot accidentally include
    # the depth decoder or the large depth tensors copied back to the CPU.
    overrides = [*args.config_override, *ACTION_ONLY_OVERRIDES]
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = baseframework.from_pretrained(
        str(args.checkpoint),
        config_overrides=overrides,
    )
    model = model.to(dtype=parameter_dtype(args.dtype)).to(device).eval()
    if hasattr(model, "set_inference_seed"):
        model.set_inference_seed(args.seed)
    if bool(getattr(model, "depth_enabled", False)):
        raise RuntimeError("depth.enabled=false override was not applied")

    # Keep checkpoint-faithful auxiliary tokens inside the Cosmos forward. They
    # can affect the visual states read by ActionDiT. Only the expensive decoder
    # and depth outputs are disabled by this action-only benchmark.
    internal_depth_tokens_enabled = bool(
        getattr(model.backbone, "use_internal_depth_tokens", False)
    )

    horizon, action_steps = validate_model_contract(
        model,
        expected_horizon=args.expected_action_horizon,
        expected_steps=args.expected_action_steps,
    )
    frame = load_current_frame(args.image)
    # The image is loaded once and remains in host memory. predict_action still
    # performs the repository's normal image resize/packing and VAE encode.
    example = {"image": [frame], "lang": args.prompt}

    torch.cuda.reset_peak_memory_stats(device)
    cold_ms, cold_output = timed_predict(model, example, device)
    cold_actions = np.asarray(cold_output["normalized_actions"])
    expected_shape_prefix = (1, horizon)
    if cold_actions.ndim != 3 or cold_actions.shape[:2] != expected_shape_prefix:
        raise RuntimeError(
            f"Unexpected normalized action shape {cold_actions.shape}; "
            f"expected [1, {horizon}, action_dim]."
        )
    unexpected_depth_keys = sorted(key for key in cold_output if "depth" in key)
    if unexpected_depth_keys:
        raise RuntimeError(
            "Action-only benchmark unexpectedly produced depth outputs: "
            f"{unexpected_depth_keys}"
        )

    print(
        f"Cold call: {cold_ms:.3f} ms; warming up for {args.warmup} calls ...",
        flush=True,
    )
    for _ in range(args.warmup):
        model.predict_action(examples=[example])
    synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    samples_ms: list[float] = []
    output = cold_output
    for index in range(args.runs):
        elapsed_ms, output = timed_predict(model, example, device)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise RuntimeError(f"Invalid timing sample at index {index}: {elapsed_ms}")
        samples_ms.append(elapsed_ms)

    actions = np.asarray(output["normalized_actions"])
    summary = latency_summary(samples_ms)
    mean_seconds = summary["mean_ms"] / 1000.0
    result: dict[str, Any] = {
        "benchmark": "cosmos2_action_only_end_to_end",
        "timed_boundary": (
            "in-memory current RGB frame + instruction -> repository preprocessing -> "
            "T5 encode -> Cosmos VAE encode -> one Cosmos transformer forward -> "
            f"{action_steps}-step ActionDiT -> normalized action chunk on CPU"
        ),
        "excluded": [
            "checkpoint loading",
            "image file decoding",
            "depth decoder",
            "depth tensor CPU transfer",
            "environment and network latency",
        ],
        "checkpoint": str(args.checkpoint.resolve()),
        "image": str(args.image.resolve()) if args.image is not None else "synthetic_224x224",
        "prompt": args.prompt,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dtype": args.dtype,
        "warmup_calls": args.warmup,
        "measured_calls": args.runs,
        "cold_call_ms": cold_ms,
        "action_horizon": horizon,
        "action_dim": int(actions.shape[-1]),
        "action_output_shape": list(actions.shape),
        "action_dit_euler_steps": action_steps,
        "internal_depth_tokens_enabled": internal_depth_tokens_enabled,
        "depth_decoder_enabled": False,
        "latency": summary,
        "policy_calls_per_second": 1.0 / mean_seconds,
        "predicted_action_steps_per_second": horizon / mean_seconds,
        "peak_allocated_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_memory_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
    }

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved result to {args.json_output}")


if __name__ == "__main__":
    main()
