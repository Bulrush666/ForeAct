#!/usr/bin/env python3
"""Qualitative Cosmos-Predict2 depth probe on one Bridge frame.

This deliberately does not use MoGe or a supervised depth checkpoint.  It
decodes the Cosmos DiT clean-latent prediction through Cosmos' own VAE and
turns the three decoded channels into a visualization.  Cosmos was trained
for video generation, so the result is a zero-shot diagnostic, not metric or
ground-truth depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foreact.model.modules.world_model.CosmoPredict2 import (  # noqa: E402
    DEFAULT_PROMPT_TEMPLATE,
    _CosmoPredict2_Interface,
    build_prompts,
)


def read_frames(video_path: Path, start: int, count: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx < start:
                continue
            if idx >= start + count:
                break
            frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path} at index {start}")
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return frames


def save_u8(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def percentile_stretch(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("pretrained/cosmos2"))
    parser.add_argument("--dataset", type=Path, default=Path("data/bridge_orig_1.0.0_lerobot"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--camera", default="image_0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/cosmos_depth_probe_episode_000000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    chunk = args.episode // 1000
    ep_name = f"episode_{args.episode:06d}"
    video = args.dataset / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{args.camera}" / f"{ep_name}.mp4"
    parquet = args.dataset / "data" / f"chunk-{chunk:03d}" / f"{ep_name}.parquet"
    task_map = {}
    with (args.dataset / "meta/tasks.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            task_map[int(item["task_index"])] = item["task"]
    task_index = int(pd.read_parquet(parquet, columns=["task_index"]).iloc[0].task_index)
    instruction = task_map.get(task_index, "")
    prompt = build_prompts([instruction], {"prompt": {"template": args.prompt_template}})[0]
    frames = read_frames(video, args.frame, args.context_frames)

    cfg = SimpleNamespace(
        framework={"world_model": {"base_wm": str(args.model), "extract_layers": [-1]}},
        prompt={"template": args.prompt_template},
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backbone = _CosmoPredict2_Interface(config=cfg).to(device).eval()
    inputs = backbone.build_inputs(images=[frames], instructions=[instruction])
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = backbone(**inputs)
        decoded = backbone.decode_depth_probe(outputs.sample).float().cpu()

    # [B, 3, T, H, W] -> the requested/current frame.  Channel averaging is
    # the same simple visualization convention used by the v6 depth probe.
    decoded_rgb = decoded[0, :, 0].permute(1, 2, 0).numpy()
    depth = decoded_rgb.mean(axis=2)
    stretched = percentile_stretch(depth)
    args.output.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output / "rgb_input.png")
    save_u8(args.output / "cosmos_decoded_rgb.png", np.clip((decoded_rgb + 1) * 127.5, 0, 255))
    save_u8(args.output / "cosmos_depth_gray.png", stretched * 255)
    import matplotlib.cm as cm

    save_u8(args.output / "cosmos_depth_color.png", cm.get_cmap("turbo")(stretched)[..., :3] * 255)
    np.save(args.output / "cosmos_depth.npy", depth.astype(np.float32))
    stats = {
        "episode": args.episode,
        "frame": args.frame,
        "camera": args.camera,
        "instruction": instruction,
        "prompt": prompt,
        "model": str(args.model),
        "context_frames": args.context_frames,
        "decoded_shape": list(decoded.shape),
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
        "depth_mean": float(np.nanmean(depth)),
        "depth_kind": "zero_shot_cosmos_vae_probe_not_metric_depth",
        "latent_interpretation": "Cosmos DiT sample (clean-latent estimate), official VAE inverse normalization",
    }
    (args.output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"Saved Cosmos depth probe to {args.output}")


if __name__ == "__main__":
    main()
