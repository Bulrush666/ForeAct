# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the terms in the repository LICENSE file.
"""
CosmoPredict2-GR00T Framework — Diffusion World Model for Action Prediction.

Uses Cosmos-Predict2 (DiT-based video world model) as the perception backbone
instead of a VLM. The DiT's intermediate representations encode rich
spatiotemporal and physical dynamics, which are projected to the action head
for continuous action prediction via flow-matching.

Architecture:
  T5 (text) + VAE (image) → DiT Transformer
    ├─ dense visual hidden states [B, 720, 2048] → ActionDiT
      └─ eight full-attention depth tokens → multi-patch head → frozen VAE decoder

Key differences from VLM4A frameworks:
  - Vision encoding: VAE latent (not pixel tokens)
  - Text encoding: T5 (not Qwen tokenizer)
  - Feature dim: 2048 (DiT) vs 2048 (Qwen-VL)
  - Features are spatiotemporal patches, not sequential tokens
"""

import math
import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from foreact.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from foreact.model.framework.base_framework import baseframework
from foreact.model.framework.share_tools import merge_framework_config
from foreact.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from foreact.model.modules.world_model import get_world_model
from foreact.model.tools import FRAMEWORK_REGISTRY
from foreact.training.trainer_utils.depth_utils import (
    depth_gradient_loss,
    depth_latent_loss,
    depth_pixel_loss,
)
from foreact.dataloader.gr00t_lerobot.image_preprocessing import (
    preprocess_observation_images,
)


@dataclass
class CosmoPredict2GR00TDefaultConfig:
    """CosmoPredict2-GR00T default parameters."""

    name: str = "CosmoPredict2GR00T"

    # === World Model backbone (Cosmos-Predict2) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "pretrained/cosmos2",
            # Which DiT layers to extract features from (-1 = last block)
            "extract_layers": [-1],
            "internal_depth_tokens": {
                "enabled": False,
                "num_tokens": 8,
                "position_mode": "global",
                "head_dim": 256,
                "coord_dim": 128,
                "seed_grid": [10, 18],
                "num_res_blocks": 2,
            },
        }
    )

    # Legacy compat: factory functions (vlm/__init__, world_model/__init__)
    # fall back to qwenvl.base_vlm when world_model.base_wm is absent.
    # vl_hidden_dim is read by some action heads (VLA_AdapterHeader, LayerwiseFM).
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "pretrained/cosmos2",
            "vl_hidden_dim": 2048,
        }
    )

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "future_action_window_size": 7,
            "action_horizon": 8,
            "past_action_window_size": 0,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                # Will be set at runtime to match world model hidden_size (2048)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("CosmoPredict2GR00T")
class CosmoPredict2_GR00T(baseframework):
    """
    World-Model-for-Action framework using Cosmos-Predict2 backbone.

    Components:
      - Cosmos-Predict2 DiT (T5 + VAE + Transformer) for spatiotemporal features
      - Flow-matching (DiT) diffusion head for continuous action prediction

    The Cosmos-Predict2 world model provides physics-aware spatiotemporal
    representations learned from large-scale video generation pretraining.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmoPredict2GR00TDefaultConfig, config)

        # Load world model backbone
        self.backbone = get_world_model(config=self.config)

        # Align cross-attention dim to world model hidden size (2048).
        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        # ActionDiT consumes Cosmos' dense final visual states. Depth tokens are
        # exposed separately, but full self-attention lets them influence the
        # dense visual states inside the Cosmos backbone.
        self.action_token_norm = nn.Identity()

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self._action_generator: Optional[torch.Generator] = None
        self._vae_generator: Optional[torch.Generator] = None
        self.inference_seed: Optional[int] = None

        self.depth_config = self.config.get("depth", {})
        self.depth_enabled = bool(self.depth_config.get("enabled", False))
        self.depth_supervision_enabled = bool(
            self.depth_config.get("supervision_enabled", False)
        )

    def _depth_config_value(self, key: str, default):
        return self.depth_config.get(key, default)

    @property
    def depth_output_kind(self) -> str:
        normalization = self._depth_config_value(
            "normalization", "genception_median_log"
        )
        return "relative_depth" if normalization == "genception_median_log" else "metric_depth_m"

    def _normalized_depth_to_output(
        self, normalized_log_depth: torch.Tensor
    ) -> torch.Tensor:
        normalized = normalized_log_depth.float().clamp(-1, 1)
        normalization = self._depth_config_value(
            "normalization", "genception_median_log"
        )
        if normalization == "genception_median_log":
            target_01 = (normalized + 1) * 0.5
            log_alpha = float(self._depth_config_value("log_alpha", 0.8613366))
            return torch.expm1(target_01 / log_alpha)
        if normalization != "metric_log":
            raise ValueError(f"Unsupported depth normalization: {normalization}")
        log_min = math.log(float(self._depth_config_value("min_depth_m", 0.05)))
        log_max = math.log(float(self._depth_config_value("max_depth_m", 10.0)))
        log_depth = (normalized + 1) * 0.5 * (log_max - log_min) + log_min
        return log_depth.exp()

    @staticmethod
    def _stack_depth_field(
        examples: List[dict], key: str, device: torch.device, dtype=None
    ) -> torch.Tensor:
        missing = [index for index, example in enumerate(examples) if key not in example]
        if missing:
            raise KeyError(
                f"Depth supervision is enabled but '{key}' is missing from "
                f"examples {missing[:8]}"
            )
        values = torch.stack([torch.as_tensor(example[key]) for example in examples])
        return values.to(device=device, dtype=dtype)

    def set_inference_seed(self, seed: Optional[int]) -> None:
        """Use independent deterministic RNG streams for VAE and action noise."""
        self.inference_seed = None if seed is None else int(seed)
        if self.inference_seed is None:
            self._action_generator = None
            self._vae_generator = None
            return

        device = next(self.action_model.parameters()).device
        self._action_generator = torch.Generator(device=device)
        self._action_generator.manual_seed(self.inference_seed)
        self._vae_generator = torch.Generator(device=device)
        self._vae_generator.manual_seed(self.inference_seed + 1)

    def _action_conditioning_features(self, wm_outputs) -> torch.Tensor:
        """Return only the dense visual states consumed by the baseline ActionDiT."""
        features = wm_outputs.hidden_states[-1]
        if features.ndim != 3:
            raise ValueError(
                f"Expected dense Cosmos features [B,N,C], got {tuple(features.shape)}"
            )
        return features

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # Step 1: World model input encoding (VAE + T5)
        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)

        # Step 2: DiT forward to extract spatiotemporal features
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            # Dense final visual states retain the baseline shape and role. With
            # full self-attention, their values may incorporate information from
            # the eight tokens carried on wm_outputs.depth_token_features.
            last_hidden = self._action_conditioning_features(wm_outputs)

        # Step 3: Action head forward and loss
        # Cosmos runs in BF16, while the freshly initialized ActionDiT is FP32
        # unless the launcher explicitly casts it. CUDA autocast does not
        # support float32 as an autocast target, so make the interface dtype
        # conversion explicit. Tensor.to(dtype) preserves the gradient path
        # from the action loss back into Cosmos.
        action_dtype = next(self.action_model.parameters()).dtype
        last_hidden = last_hidden.to(dtype=action_dtype)
        with torch.autocast("cuda", enabled=False):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=action_dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=action_dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        depth_latents = wm_outputs.depth_latents if self.depth_enabled else None
        if self.depth_enabled and depth_latents is None:
            raise RuntimeError(
                "depth.enabled=true requires world_model.internal_depth_tokens.enabled=true"
            )

        depth_target = None
        depth_valid_mask = None
        depth_frame_mask = None
        depth_target_latents = None
        if self.depth_supervision_enabled:
            depth_target = self._stack_depth_field(
                examples, "depth_target", last_hidden.device, torch.float32
            )
            depth_valid_mask = self._stack_depth_field(
                examples, "depth_valid_mask", last_hidden.device, torch.bool
            )
            depth_frame_mask = self._stack_depth_field(
                examples, "depth_frame_mask", last_hidden.device, torch.bool
            )
            depth_target_latents = self.backbone.encode_depth_target(depth_target)

        latent_weight = float(self._depth_config_value("latent_loss_weight", 0.0))
        pixel_weight = float(self._depth_config_value("pixel_loss_weight", 0.0))
        gradient_weight = float(
            self._depth_config_value("gradient_loss_weight", 0.0)
        )
        requires_decoder_loss = self.depth_supervision_enabled and (
            pixel_weight > 0 or gradient_weight > 0
        )
        should_decode_depth = depth_latents is not None and (
            bool(self._depth_config_value("decode_in_train", False))
            or requires_decoder_loss
        )
        depth_rgb_video = (
            self.backbone.decode_depth(depth_latents) if should_decode_depth else None
        )
        normalized_log_depth_video = (
            depth_rgb_video.mean(dim=1, keepdim=True)
            if depth_rgb_video is not None
            else None
        )
        depth_video = (
            self._normalized_depth_to_output(normalized_log_depth_video)
            if normalized_log_depth_video is not None
            else None
        )

        current_depth_latent_loss = None
        if (
            depth_latents is not None
            and depth_target_latents is not None
            and latent_weight > 0
        ):
            current_depth_latent_loss = depth_latent_loss(
                prediction=depth_latents,
                target=depth_target_latents,
                frame_mask=depth_frame_mask,
            )

        current_depth_pixel_loss = None
        current_depth_gradient_loss = None
        if (
            depth_rgb_video is not None
            and depth_target is not None
            and depth_valid_mask is not None
        ):
            depth_target_rgb = depth_target.expand(
                -1, depth_rgb_video.shape[1], -1, -1, -1
            )
            depth_valid_mask_rgb = depth_valid_mask.expand_as(depth_target_rgb)
            if pixel_weight > 0:
                current_depth_pixel_loss = depth_pixel_loss(
                    depth_rgb_video, depth_target_rgb, depth_valid_mask_rgb
                )
            if gradient_weight > 0:
                current_depth_gradient_loss = depth_gradient_loss(
                    depth_rgb_video, depth_target_rgb, depth_valid_mask_rgb
                )

        weighted_depth_losses = []
        for weight, loss in (
            (latent_weight, current_depth_latent_loss),
            (pixel_weight, current_depth_pixel_loss),
            (gradient_weight, current_depth_gradient_loss),
        ):
            if loss is not None:
                weighted_depth_losses.append(weight * loss)
        depth_loss = (
            torch.stack(weighted_depth_losses).sum()
            if weighted_depth_losses
            else None
        )
        total_loss = float(
            self._depth_config_value("action_loss_weight", 1.0)
        ) * action_loss
        if depth_loss is not None:
            total_loss = total_loss + float(
                self._depth_config_value("loss_weight", 0.01)
            ) * depth_loss

        return {
            "action_loss": action_loss,
            "depth_loss": depth_loss,
            "depth_latent_loss": current_depth_latent_loss,
            "depth_pixel_loss": current_depth_pixel_loss,
            "depth_gradient_loss": current_depth_gradient_loss,
            "total_loss": total_loss,
            "depth_latents": depth_latents,
            "depth_token_features": wm_outputs.depth_token_features,
            "depth_rgb_video": depth_rgb_video,
            "normalized_log_depth_video": normalized_log_depth_video,
            "depth_video": depth_video,
            "depth_output_kind": self.depth_output_kind,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        image_cfg = self.config.datasets.vla_data
        crop_xyxy = getattr(image_cfg, "obs_image_crop_xyxy", None)
        size_hw = getattr(image_cfg, "obs_image_size_hw", None)
        if size_hw is None:
            # Compatibility for older square configs.  The legacy field was
            # passed directly to PIL and is therefore safe only when H == W.
            legacy_size = getattr(image_cfg, "obs_image_size", None)
            if legacy_size and int(legacy_size[0]) != int(legacy_size[1]):
                raise ValueError(
                    "non-square obs_image_size is ambiguous; use obs_image_size_hw"
                )
            size_hw = legacy_size
        if crop_xyxy is not None or size_hw is not None:
            batch_images = preprocess_observation_images(
                batch_images,
                crop_xyxy=crop_xyxy,
                size_hw=size_hw,
            )

        wm_inputs = self.backbone.build_inputs(
            images=batch_images,
            instructions=instructions,
            generator=self._vae_generator,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = self._action_conditioning_features(wm_outputs)

        action_dtype = next(self.action_model.parameters()).dtype
        last_hidden = last_hidden.to(dtype=action_dtype)
        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=action_dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", enabled=False):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                generator=self._action_generator,
            )

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        output = {"normalized_actions": normalized_actions}
        if self.depth_enabled:
            depth_latents = wm_outputs.depth_latents
            if depth_latents is None:
                raise RuntimeError(
                    "depth.enabled=true requires world_model.internal_depth_tokens.enabled=true"
                )
            depth_rgb_video = self.backbone.decode_depth(depth_latents)
            normalized_log_depth_video = depth_rgb_video.mean(dim=1, keepdim=True)
            depth_video = self._normalized_depth_to_output(
                normalized_log_depth_video
            )
            output.update(
                {
                    "depth_latents": depth_latents.detach().float().cpu().numpy(),
                    "depth_rgb_video": depth_rgb_video.detach().cpu().numpy(),
                    "normalized_log_depth_video": normalized_log_depth_video.detach()
                    .cpu()
                    .numpy(),
                    "depth_video": depth_video.detach().cpu().numpy(),
                    "depth_output_kind": self.depth_output_kind,
                }
            )
        return output


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="configs/train/libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "CosmoPredict2GR00T"
    cfg.framework.world_model = {
        "base_wm": "pretrained/cosmos2",
        "extract_layers": [-1],
    }

    model: CosmoPredict2_GR00T = CosmoPredict2_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
