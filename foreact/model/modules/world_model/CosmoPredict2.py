# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the terms in the repository LICENSE file.
"""
Cosmos-Predict2 World Model Interface.

Wraps NVIDIA Cosmos-Predict2 (diffusion-based Video2World model) as a
world-model backend for foreact action prediction frameworks.

Architecture:
  - T5EncoderModel: text instruction → text embeddings [B, L_text, 1024]
  - AutoencoderKLWan (VAE): observation images → video latents [B, C, T, H, W]
  - CosmosTransformer3DModel (DiT): 28-layer transformer, hidden_dim=2048
    Takes noised latents + text embeddings → denoised latents
    We extract intermediate hidden states for action-conditioning.

Key difference from VLM wrappers:
  - No chat template / processor — uses T5 for text, VAE for vision
  - Hidden states come from DiT blocks, not autoregressive LM
  - The `build_inputs` interface provides a clean world-model API
    that does not depend on VLM-specific naming conventions.
"""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreact.training.trainer_utils import initialize_overwatch
from foreact.model.modules.depth_decoder import DepthVideoDecoder

logger = initialize_overwatch(__name__)


class _SpatialResidualBlock(nn.Module):
    """Batch-size-independent residual refinement for a 2-D latent feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        num_groups = min(32, channels)
        while channels % num_groups:
            num_groups -= 1
        self.norm1 = nn.GroupNorm(num_groups, channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels, eps=1e-6)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        return residual + hidden_states


class GlobalTokenSeedDepthHead(nn.Module):
    """Jointly decode global depth tokens into the native Cosmos patch grid.

    All depth tokens contribute to every seed location.  No token owns a row,
    column, quadrant, or other fixed output region.  Global UV coordinates give
    the shared seed MLP an output location, after which ordinary 2-D residual
    convolutions enforce spatial continuity before native ``unpatchify``.  An
    optional zero-initialized spatial skip projects the frozen image-VAE latent
    onto the same seed grid and adds it as a learned residual correction.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_dim: int,
        num_tokens: int,
        head_dim: int = 256,
        coord_dim: int = 128,
        seed_grid: tuple[int, int] = (10, 18),
        num_res_blocks: int = 2,
        use_vae_seed_fusion: bool = False,
        vae_channels: int = 16,
        vae_downsample_factor: int = 4,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if head_dim <= 0 or coord_dim <= 0:
            raise ValueError(
                f"head_dim and coord_dim must be positive, got {head_dim}, {coord_dim}"
            )
        seed_rows, seed_cols = (int(seed_grid[0]), int(seed_grid[1]))
        if seed_rows <= 0 or seed_cols <= 0:
            raise ValueError(f"seed_grid must be positive, got {seed_grid}")
        if num_res_blocks < 0:
            raise ValueError(
                f"num_res_blocks must be non-negative, got {num_res_blocks}"
            )
        if use_vae_seed_fusion and vae_channels <= 0:
            raise ValueError(f"vae_channels must be positive, got {vae_channels}")
        if use_vae_seed_fusion and vae_downsample_factor <= 0:
            raise ValueError(
                "vae_downsample_factor must be positive, got "
                f"{vae_downsample_factor}"
            )

        self.num_tokens = int(num_tokens)
        self.head_dim = int(head_dim)
        self.patch_dim = int(patch_dim)
        self.seed_grid = (seed_rows, seed_cols)
        self.use_vae_seed_fusion = bool(use_vae_seed_fusion)
        self.vae_channels = int(vae_channels)
        self.vae_downsample_factor = int(vae_downsample_factor)
        self.token_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.token_proj = nn.Linear(hidden_size, self.head_dim)
        self.coordinate_proj = nn.Sequential(
            nn.Linear(2, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
        )

        context_dim = self.num_tokens * self.head_dim
        self.seed_mlp = nn.Sequential(
            nn.LayerNorm(context_dim + coord_dim, eps=1e-6),
            nn.Linear(context_dim + coord_dim, 2 * self.head_dim),
            nn.SiLU(),
            nn.Linear(2 * self.head_dim, self.head_dim),
        )
        self.seed_block = _SpatialResidualBlock(self.head_dim)

        if self.use_vae_seed_fusion:
            num_groups = min(32, self.head_dim)
            while self.head_dim % num_groups:
                num_groups -= 1
            self.vae_seed_proj = nn.Sequential(
                nn.Conv2d(
                    self.vae_channels,
                    self.head_dim,
                    kernel_size=self.vae_downsample_factor,
                    stride=self.vae_downsample_factor,
                ),
                nn.GroupNorm(num_groups, self.head_dim, eps=1e-6),
                nn.SiLU(),
            )
            # A zero convolution makes the initial fused head exactly identical
            # to the query-only head.  It learns the VAE-to-query channel basis
            # before the spatial skip can influence the predicted depth latent.
            self.vae_seed_out = nn.Conv2d(
                self.head_dim, self.head_dim, kernel_size=1
            )
            nn.init.zeros_(self.vae_seed_out.weight)
            nn.init.zeros_(self.vae_seed_out.bias)
        else:
            self.vae_seed_proj = None
            self.vae_seed_out = None

        self.output_blocks = nn.ModuleList(
            [_SpatialResidualBlock(self.head_dim) for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Conv2d(
            self.head_dim, self.patch_dim, kernel_size=1
        )

        # Keep the initial frozen-VAE input small without severing the first-step
        # gradient path into the joint token context.
        nn.init.normal_(self.output_proj.weight, std=1.0e-3)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _normalized_centers(
        length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return (
            (torch.arange(length, device=device, dtype=dtype) + 0.5)
            .div(length)
            .mul(2)
            .sub(1)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        output_grid: tuple[int, int],
        vae_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,Q,D], got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"expected {self.num_tokens} global tokens, got {tokens.shape[1]}"
            )
        output_rows, output_cols = (int(output_grid[0]), int(output_grid[1]))
        if output_rows <= 0 or output_cols <= 0:
            raise ValueError(f"output_grid must be positive, got {output_grid}")

        token_features = self.token_proj(self.token_norm(tokens))
        token_context = token_features.flatten(1)

        seed_rows, seed_cols = self.seed_grid
        rows = self._normalized_centers(seed_rows, tokens.device, tokens.dtype)
        cols = self._normalized_centers(seed_cols, tokens.device, tokens.dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        global_coordinates = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
        coordinate_features = self.coordinate_proj(global_coordinates)

        num_seed_positions = seed_rows * seed_cols
        expanded_context = token_context.unsqueeze(1).expand(
            -1, num_seed_positions, -1
        )
        expanded_coordinates = coordinate_features.unsqueeze(0).expand(
            tokens.shape[0], -1, -1
        )
        seed_features = self.seed_mlp(
            torch.cat((expanded_context, expanded_coordinates), dim=-1)
        )
        hidden_states = seed_features.transpose(1, 2).reshape(
            tokens.shape[0], self.head_dim, seed_rows, seed_cols
        )
        hidden_states = self.seed_block(hidden_states)

        if self.use_vae_seed_fusion:
            if vae_latents is None:
                raise ValueError(
                    "vae_latents are required when VAE seed fusion is enabled"
                )
            if vae_latents.ndim != 5:
                raise ValueError(
                    "vae_latents must be [B,C,T,H,W], got "
                    f"{tuple(vae_latents.shape)}"
                )
            if vae_latents.shape[0] != tokens.shape[0]:
                raise ValueError(
                    "tokens and vae_latents must have the same batch size, got "
                    f"{tokens.shape[0]} and {vae_latents.shape[0]}"
                )
            if vae_latents.shape[1] != self.vae_channels:
                raise ValueError(
                    f"expected {self.vae_channels} VAE channels, got "
                    f"{vae_latents.shape[1]}"
                )
            if vae_latents.shape[2] != 1:
                raise ValueError(
                    "VAE seed fusion predicts one target frame and requires one "
                    f"input latent frame, got {vae_latents.shape[2]}"
                )

            vae_seed = self.vae_seed_proj(vae_latents[:, :, 0])
            if vae_seed.shape[-2:] != (seed_rows, seed_cols):
                vae_seed = F.interpolate(
                    vae_seed,
                    size=(seed_rows, seed_cols),
                    mode="bilinear",
                    align_corners=False,
                )
            vae_residual = self.vae_seed_out(vae_seed).to(hidden_states.dtype)
            hidden_states = hidden_states + vae_residual

        if (seed_rows, seed_cols) != (output_rows, output_cols):
            hidden_states = F.interpolate(
                hidden_states,
                size=(output_rows, output_cols),
                mode="bilinear",
                align_corners=False,
            )
        for block in self.output_blocks:
            hidden_states = block(hidden_states)

        patch_map = self.output_proj(hidden_states)
        return patch_map.permute(0, 2, 3, 1).reshape(
            tokens.shape[0], output_rows * output_cols, self.patch_dim
        )


class CosmosDepthDecoder:
    """Decode a Cosmos denoised latent prediction for a depth probe.

    Cosmos-Predict2 is a video-generation model, not a depth-estimation model.
    Its DiT output is an estimate of the clean VAE latent (``sample`` in
    diffusers), so this module only performs the official inverse latent
    normalization followed by VAE decoding.  The resulting image is a
    *zero-shot probe* and must not be interpreted as metric depth without a
    separately trained depth head/loss.
    """

    def __init__(self, vae: nn.Module, scheduler) -> None:
        self.vae = vae
        self.scheduler = scheduler

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        config = self.vae.config
        if getattr(config, "latents_mean", None) is None:
            return latents
        mean = torch.as_tensor(config.latents_mean, device=latents.device, dtype=latents.dtype)
        std = torch.as_tensor(config.latents_std, device=latents.device, dtype=latents.dtype)
        mean = mean.view(1, config.z_dim, 1, 1, 1)
        std = std.view(1, config.z_dim, 1, 1, 1)
        return latents * std / self.scheduler.config.sigma_data + mean

    @torch.inference_mode()
    def forward(self, clean_latents: torch.Tensor) -> torch.Tensor:
        """Return decoded video in VAE range ``[-1, 1]``.

        ``clean_latents`` is expected to be the Cosmos DiT ``sample`` output,
        still in the scheduler-normalized latent space.
        """
        latents = self._denormalize_latents(clean_latents)
        return self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]

    __call__ = forward


# Keep the text contract identical to the current Cosmos2 training config. The
# frozen T5 encoder sees this fully rendered string rather than the raw
# LeRobot language field. Explicitly naming the next observation frame keeps
# the prompt aligned with the t+1 depth target.
DEFAULT_PROMPT_TEMPLATE = (
    "Robot task: {instruction}. Predict the depth of the next observation "
    "frame and the next 16 robot actions."
)


def build_prompts(instructions, config=None):
    """Render and validate prompts using the current Cosmos2 convention.

    The wrapper normalizes arbitrary instruction whitespace and rejects empty
    language fields before applying ``prompt.template``. Keeping this in the Cosmos
    world-model input path makes training and server-side evaluation identical,
    including for checkpoints whose older config did not contain ``prompt``.
    """

    template = DEFAULT_PROMPT_TEMPLATE
    if config is not None:
        try:
            prompt_cfg = config.get("prompt", None)
        except AttributeError:
            prompt_cfg = getattr(config, "prompt", None)
        if prompt_cfg is not None:
            try:
                configured_template = prompt_cfg.get("template", None)
            except AttributeError:
                configured_template = getattr(prompt_cfg, "template", None)
            if configured_template is not None:
                template = str(configured_template)

    if "{instruction}" not in template:
        raise ValueError("prompt.template must contain {instruction}")

    prompts = []
    for instruction in instructions:
        normalized = " ".join(str(instruction).strip().split())
        if not normalized:
            raise ValueError("Robot instruction cannot be empty")
        try:
            prompts.append(template.format(instruction=normalized))
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                "prompt.template must be a valid Python format string using "
                "the {instruction} field"
            ) from error
    return prompts


class _CosmoPredict2_Interface(nn.Module):
    """
    World model wrapper for Cosmos-Predict2 (diffusers-based).

    Exposes a compatible interface with VLM wrappers so that framework
    code can swap VLM ↔ WM transparently. The key methods are:
      - forward(**kwargs) → model outputs with hidden_states
      - build_inputs(images, instructions) → dict of tensors
      - generate(**kwargs) → video generation (optional)

    Representation extraction strategy:
      We run a single DiT forward pass at noise level σ≈0 and register
      forward hooks to capture intermediate block outputs. These are
      concatenated/pooled to produce a [B, N_tokens, hidden_dim] tensor
      that the action head can consume — analogous to VLM hidden_states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", "nvidia/Cosmos-Predict2-2B-Video2World"),
        )
        self.config = config

        # Import diffusers components
        from diffusers import (
            AutoencoderKLWan,
            CosmosTransformer3DModel,
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import T5EncoderModel, T5TokenizerFast

        logger.info(f"Loading Cosmos-Predict2 from {model_name}")

        # Load components individually (Pipeline is not nn.Module; split loading enables per-component freeze/finetune)
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        # Optional, frozen decoder used by the standalone zero-shot depth
        # probe.  It adds no trainable parameters and is not part of the
        # action-training loss.
        depth_cfg = config.get("depth", {})
        self.depth_decoder = DepthVideoDecoder(
            self.vae,
            self.scheduler,
            gradient_checkpointing=bool(
                depth_cfg.get("decoder_gradient_checkpointing", True)
            ),
        )

        # Use diffusers' VideoProcessor for image/video preprocessing (resize, normalize, etc.)
        from diffusers.video_processor import VideoProcessor
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Freeze VAE and text encoder by default
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Expose config compatible with framework expectations
        # DiT: 16 heads × 128 dim = 2048
        self._hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        # Optional sparse depth tokens live inside the Cosmos token sequence.
        # They are appended after patch embedding (not to the 5-D VAE latent),
        # participate in all native Cosmos self-/cross-attention blocks, and are
        # removed again before Cosmos' output projection/unpatchify path.
        token_cfg = wm_cfg.get(
            "internal_depth_tokens", wm_cfg.get("internal_action_tokens", {})
        )
        self.use_internal_depth_tokens = bool(token_cfg.get("enabled", False))
        # Compatibility alias for older utilities; these tokens no longer feed ActionDiT.
        self.use_internal_action_tokens = self.use_internal_depth_tokens
        self.num_internal_depth_tokens = int(token_cfg.get("num_tokens", 8))
        position_mode = str(token_cfg.get("position_mode", "global"))
        if position_mode != "global":
            raise ValueError(
                "world_model.internal_depth_tokens.position_mode must be 'global', "
                f"got {position_mode!r}"
            )
        seed_grid = tuple(token_cfg.get("seed_grid", (10, 18)))
        if len(seed_grid) != 2:
            raise ValueError(
                "world_model.internal_depth_tokens.seed_grid must contain two integers"
            )
        vae_seed_cfg = token_cfg.get("vae_seed_fusion", {})

        if self.use_internal_depth_tokens:
            self.learnable_depth_tokens = nn.Parameter(
                torch.empty(1, self.num_internal_depth_tokens, self._hidden_size)
            )
            nn.init.normal_(self.learnable_depth_tokens, std=self._hidden_size**-0.5)

            self.depth_head = GlobalTokenSeedDepthHead(
                hidden_size=self._hidden_size,
                patch_dim=int(self.transformer.proj_out.out_features),
                num_tokens=self.num_internal_depth_tokens,
                head_dim=int(token_cfg.get("head_dim", 256)),
                coord_dim=int(token_cfg.get("coord_dim", 128)),
                seed_grid=(int(seed_grid[0]), int(seed_grid[1])),
                num_res_blocks=int(token_cfg.get("num_res_blocks", 2)),
                use_vae_seed_fusion=bool(vae_seed_cfg.get("enabled", False)),
                vae_channels=int(
                    vae_seed_cfg.get(
                        "input_channels", getattr(self.vae.config, "z_dim", 16)
                    )
                ),
                vae_downsample_factor=int(
                    vae_seed_cfg.get("downsample_factor", 4)
                ),
            )
        else:
            self.register_parameter("learnable_depth_tokens", None)
            self.depth_head = None

        self.depth_backbone_gradient_scale = float(
            depth_cfg.get("backbone_gradient_scale", 0.01)
        )

        # Create a config-like object for the framework to read hidden_size
        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # Hook storage for intermediate features
        self._intermediate_features = []
        self._hooks = []

        # Which transformer blocks to extract features from (-1 = last)
        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        # The legacy path needs hooks to expose a selected visual layer.  The
        # internal-token path directly returns the final sparse-token states.
        if not self.use_internal_depth_tokens:
            self._register_hooks()

    @property
    def model(self):
        """Compatibility shim: framework code accesses self.qwen_vl_interface.model.config.hidden_size"""
        class _ModelShim:
            pass
        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _register_hooks(self):
        """Register forward hooks on selected transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.transformer_blocks)
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                block = self.transformer.transformer_blocks[actual_idx]
                hook = block.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        """Capture intermediate transformer block output."""
        # DiT block output is a tuple; first element is hidden_states
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _depth_token_rotary_emb(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return identity RoPE for global tokens with no spatial ownership."""
        rope = self.transformer.rope
        device = hidden_states.device
        rotary_dim = int(rope.dim_t + rope.dim_h + rope.dim_w)
        shape = (self.num_internal_depth_tokens, rotary_dim)
        return (
            torch.ones(shape, device=device, dtype=torch.float32),
            torch.zeros(shape, device=device, dtype=torch.float32),
        )

    @staticmethod
    def _forward_block_with_depth_tokens(
        block,
        hidden_states,
        encoder_hidden_states,
        embedded_timestep,
        temb,
        image_rotary_emb,
        extra_pos_emb,
        attention_mask,
        controlnet_residual,
    ):
        """Cosmos block with visual and depth tokens in full self-attention."""
        if block.before_proj is not None:
            raise NotImplementedError("Depth observer tokens do not support before_proj")
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb
        norm_hidden_states, gate = block.norm1(hidden_states, embedded_timestep, temb)
        attn_output = block.attn1(
            norm_hidden_states,
            attention_mask=None,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + gate * attn_output
        norm_hidden_states, gate = block.norm2(hidden_states, embedded_timestep, temb)
        attn_output = block.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + gate * attn_output
        norm_hidden_states, gate = block.norm3(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * block.ff(norm_hidden_states)
        if controlnet_residual is not None:
            hidden_states = hidden_states + controlnet_residual
        if block.after_proj is not None:
            raise NotImplementedError("Depth observer tokens do not support after_proj")
        return hidden_states

    def _forward_with_internal_depth_tokens(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        block_controlnet_hidden_states=None,
        attention_mask: torch.Tensor | None = None,
        fps: int | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run visual and sparse depth tokens with full attention in all blocks."""
        transformer = self.transformer
        batch_size, _, num_frames, height, width = hidden_states.shape
        # Preserve the normalized, image-only VAE latent before Cosmos appends
        # condition and padding channels.  The frozen VAE is not run a second
        # time, so both the transformer and the depth skip see the same sample.
        image_vae_latents = hidden_states

        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if transformer.config.concat_padding_mask:
            if padding_mask is None:
                padding_mask = hidden_states.new_zeros(1, 1, height, width)
            padding_mask_resized = F.interpolate(
                padding_mask, size=list(hidden_states.shape[-2:]), mode="nearest"
            )
            padding_channel = padding_mask_resized.unsqueeze(2).repeat(
                batch_size, 1, num_frames, 1, 1
            )
            hidden_states = torch.cat([hidden_states, padding_channel], dim=1)

        if attention_mask is not None:
            # Tokenizers return int64 masks, while PyTorch SDPA requires a
            # bool/float mask (or the same dtype as Q/K). Cosmos expects True
            # for valid text positions.
            attention_mask = attention_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)

        visual_rotary_emb = transformer.rope(hidden_states, fps=fps)
        depth_rotary_emb = self._depth_token_rotary_emb(hidden_states)
        image_rotary_emb = tuple(
            torch.cat((visual, depth), dim=0)
            for visual, depth in zip(visual_rotary_emb, depth_rotary_emb)
        )

        visual_extra_pos_emb = (
            transformer.learnable_pos_embed(hidden_states)
            if transformer.config.extra_pos_embed_type
            else None
        )

        patch_t, patch_h, patch_w = transformer.config.patch_size
        post_patch_num_frames = num_frames // patch_t
        post_patch_height = height // patch_h
        post_patch_width = width // patch_w

        hidden_states = transformer.patch_embed(hidden_states).flatten(1, 3)
        num_visual_tokens = hidden_states.shape[1]
        depth_tokens = self.learnable_depth_tokens.to(dtype=hidden_states.dtype).expand(
            batch_size, -1, -1
        )
        hidden_states = torch.cat((hidden_states, depth_tokens), dim=1)

        extra_pos_emb = visual_extra_pos_emb
        if extra_pos_emb is not None:
            depth_extra_pos = extra_pos_emb.new_zeros(
                batch_size, self.num_internal_depth_tokens, extra_pos_emb.shape[-1]
            )
            extra_pos_emb = torch.cat((extra_pos_emb, depth_extra_pos), dim=1)

        if timestep.ndim == 1:
            temb, embedded_timestep = transformer.time_embed(hidden_states, timestep)
            output_temb = temb
            output_embedded_timestep = embedded_timestep
        elif timestep.ndim == 5:
            expected_shape = (batch_size, 1, num_frames, 1, 1)
            if timestep.shape != expected_shape:
                raise ValueError(f"Expected timestep shape {expected_shape}, got {timestep.shape}")
            timestep = timestep.flatten()
            visual_temb, visual_embedded_timestep = transformer.time_embed(hidden_states, timestep)
            visual_temb, visual_embedded_timestep = (
                value.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for value in (visual_temb, visual_embedded_timestep)
            )
            output_temb = visual_temb
            output_embedded_timestep = visual_embedded_timestep
            temb = torch.cat(
                (
                    visual_temb,
                    visual_temb.mean(dim=1, keepdim=True).expand(
                        -1, self.num_internal_depth_tokens, -1
                    ),
                ),
                dim=1,
            )
            embedded_timestep = torch.cat(
                (
                    visual_embedded_timestep,
                    visual_embedded_timestep.mean(dim=1, keepdim=True).expand(
                        -1, self.num_internal_depth_tokens, -1
                    ),
                ),
                dim=1,
            )
        else:
            raise ValueError(
                f"Expected timestep to have shape [B] or [B, 1, T, 1, 1], got {timestep.shape}"
            )

        text_context, img_context = (
            encoder_hidden_states
            if isinstance(encoder_hidden_states, tuple)
            else (encoder_hidden_states, None)
        )
        if transformer.config.use_crossattn_projection:
            text_context = transformer.crossattn_proj(text_context)
        if img_context is not None and transformer.config.img_context_dim_in:
            img_context = transformer.img_context_proj(img_context)
        processed_encoder_hidden_states = (
            (text_context, img_context)
            if isinstance(encoder_hidden_states, tuple)
            else text_context
        )

        controlnet_block_index_map = {}
        if block_controlnet_hidden_states is not None:
            n_blocks = len(transformer.transformer_blocks)
            controlnet_block_index_map = {
                block_idx: block_controlnet_hidden_states[index]
                for index, block_idx in enumerate(
                    range(0, n_blocks, transformer.config.controlnet_block_every_n)
                )
            }

        for block_idx, block in enumerate(transformer.transformer_blocks):
            controlnet_residual = controlnet_block_index_map.get(block_idx)
            if torch.is_grad_enabled() and transformer.gradient_checkpointing:
                hidden_states = transformer._gradient_checkpointing_func(
                    partial(self._forward_block_with_depth_tokens, block),
                    hidden_states,
                    processed_encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                    controlnet_residual,
                )
            else:
                hidden_states = self._forward_block_with_depth_tokens(
                    block,
                    hidden_states,
                    processed_encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                    controlnet_residual,
                )

        # Read the sparse tokens before Cosmos' generation-only output head.
        # This matches the previous last-block hook semantics.
        depth_token_features = hidden_states[:, num_visual_tokens:]
        visual_token_features = hidden_states[:, :num_visual_tokens]

        video_hidden_states = transformer.norm_out(
            visual_token_features, output_embedded_timestep, output_temb
        )
        video_hidden_states = transformer.proj_out(video_hidden_states)

        scaled_depth_tokens = depth_token_features.detach() + self.depth_backbone_gradient_scale * (
            depth_token_features - depth_token_features.detach()
        )
        if post_patch_num_frames != 1:
            raise ValueError(
                "The sparse depth-token head predicts one target latent frame; "
                f"got a temporal patch grid of {post_patch_num_frames}"
            )
        depth_patches = self.depth_head(
            scaled_depth_tokens,
            output_grid=(post_patch_height, post_patch_width),
            vae_latents=image_vae_latents,
        )

        def unpatchify(value: torch.Tensor) -> torch.Tensor:
            value = value.unflatten(
                2, (patch_h, patch_w, patch_t, -1)
            )
            value = value.unflatten(
                1, (post_patch_num_frames, post_patch_height, post_patch_width)
            )
            value = value.permute(0, 7, 1, 6, 2, 4, 3, 5)
            return value.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        video_sample = unpatchify(video_hidden_states)
        depth_latents = unpatchify(depth_patches)
        return video_sample, visual_token_features, depth_token_features, depth_latents

    def _encode_text(self, instructions, max_length=512):
        """Encode text instructions using T5."""
        instructions = build_prompts(instructions, self.config)
        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state  # [B, L, 1024]

        return text_embeds, text_inputs.attention_mask

    def _encode_images(self, images, num_frames=None, generator=None):
        """Encode observation images through VAE to get latent tokens.

        Follows the Cosmos pipeline approach: pad images to uniform frame count
        with last-frame repetition, then VAE-encode the whole video.

        Args:
            images: List of List of PIL Images [B, [imgs...]]
            num_frames: If given, pad/truncate to this exact count.
                If None (default), pad to the max frame count in the batch.
                VAE temporal factor is 4, so T_latent = (num_frames-1)//4+1.

        Returns:
            latents: [B, C, T_latent, H/8, W/8] video latent tensor
            cond_frame_counts: list[int], real frame count per sample (before padding)
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        # 480×832 is the pretrained resolution; spatial dims must be multiples of 16.
        # Smaller sizes (e.g. 224×224) technically work but hurt quality due to positional embedding mismatch.
        # If saving VRAM, keep ~16:9 aspect ratio: 256×448 or 320×576.
        height, width = 320, 576

        # First pass: preprocess each sample, record real frame counts
        preprocessed = []
        cond_frame_counts = []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]

            video_tensor = self.video_processor.preprocess_video(sample_imgs, height=height, width=width)
            video_tensor = video_tensor.to(device=device, dtype=dtype)  # [1, C, n_imgs, H, W]
            preprocessed.append(video_tensor)
            cond_frame_counts.append(video_tensor.shape[2])

        # Determine target frame count: use num_frames if specified, otherwise batch max
        # Ensure at least 1 frame (VAE needs T >= 1)
        if num_frames is None:
            target_frames = max(cond_frame_counts)
        else:
            target_frames = num_frames

        # Second pass: truncate or pad each sample to target_frames
        batch_videos = []
        for i, video_tensor in enumerate(preprocessed):
            n = video_tensor.shape[2]
            if n > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
                cond_frame_counts[i] = target_frames
            elif n < target_frames:
                # Pad with last-frame repetition (matches official pipeline)
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))  # [C, target_frames, H, W]

        # Stack to [B, C, target_frames, H, W]
        video = torch.stack(batch_videos, dim=0)

        with torch.no_grad():
            # T_latent = temporal latent frames = (num_frames-1)//4+1  (VAE temporal downsample factor=4)
            # e.g. 5 frames → 2 latent frames, 9 frames → 3 latent frames
            latents = self.vae.encode(video).latent_dist.sample(generator=generator)

        # Normalize latents (matches official pipeline: prepare_latents) # TODO check if this normalization is actually needed
        if self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            ) 
            sigma_data = self.scheduler.config.sigma_data
            latents = (latents - latents_mean) / latents_std * sigma_data

        return latents, cond_frame_counts # latents: [B, C, T_latent, H/8, W/8], cond_frame_counts: list[int]

    def build_inputs(self, images, instructions, **kwargs):
        """Build inputs for the DiT world model.

        Instead of chat templates (VLM), we:
        1. Encode text with T5
        2. Encode images with VAE
        3. Package for the DiT forward pass

        Returns:
            dict with keys matching what forward() expects
        """
        assert len(images) == len(instructions)

        # Ensure encoders are on the right device
        device = next(self.transformer.parameters()).device
        # self.text_encoder.to(device)
        # self.vae.to(device)

        text_embeds, text_mask = self._encode_text(instructions)
        latents, cond_frame_counts = self._encode_images(
            images, generator=kwargs.pop("generator", None)
        )

        # Offload T5 and VAE to CPU to free VRAM for the transformer
        # self.text_encoder.to("cpu")
        # self.vae.to("cpu")
        # torch.cuda.empty_cache()

        # For feature extraction, use timestep=0 (clean / minimal noise)
        batch_size = latents.shape[0]
        device = latents.device
        _, _, t_lat, h_lat, w_lat = latents.shape # B, C, T_latent, H/8, W/8
        timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
        # condition_mask: tells DiT which latent frames are reliable conditions (=1) vs to-be-generated (=0).
        # In Video2World, this separates input frames from predicted future frames.
        # Here (action prediction, not generation), we set timestep=0 + condition_mask on real frames
        # so DiT runs a near-clean forward pass for feature extraction, not actual denoising.
        # in_channels = 16 (latents) + 1 (condition_mask) = 17
        # Shape: [B, 1, T_latent, H_latent, W_latent]
        condition_mask = latents.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        for i, n_cond in enumerate(cond_frame_counts):
            # Map pixel-frame count to latent-frame count
            n_cond_latent = (n_cond - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[i, :, :n_cond_latent] = 1.0

        # padding_mask: concat_padding_mask=True adds 1 more channel → 18 total
        # Shape: [1, 1, H_orig, W_orig] — all zeros = no padding
        # Will be resized to latent spatial dims by the transformer
        padding_mask = latents.new_zeros(1, 1, h_lat, w_lat)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "attention_mask": text_mask,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        """Forward pass through the DiT transformer.

        Runs a single-step forward to extract rich spatiotemporal features.
        Returns an output object with .hidden_states for compatibility.
        """
        is_wm = kwargs.pop("_is_wm_input", False)
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        return_dict = kwargs.pop("return_dict", True)
        kwargs.pop("output_attentions", None)

        # Clear feature buffer
        self._intermediate_features.clear()

        depth_token_features = None
        depth_latents = None
        visual_token_features = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.use_internal_depth_tokens:
                (
                    sample,
                    visual_token_features,
                    depth_token_features,
                    depth_latents,
                ) = self._forward_with_internal_depth_tokens(
                    hidden_states=kwargs["hidden_states"],
                    timestep=kwargs["timestep"],
                    encoder_hidden_states=kwargs["encoder_hidden_states"],
                    block_controlnet_hidden_states=kwargs.get(
                        "block_controlnet_hidden_states", None
                    ),
                    attention_mask=kwargs.get("attention_mask", None),
                    fps=kwargs.get("fps", None),
                    condition_mask=kwargs.get("condition_mask", None),
                    padding_mask=kwargs.get("padding_mask", None),
                )
                dit_output = sample
            else:
                dit_output = self.transformer(
                    hidden_states=kwargs["hidden_states"],
                    timestep=kwargs["timestep"],
                    encoder_hidden_states=kwargs["encoder_hidden_states"],
                    attention_mask=(
                        kwargs["attention_mask"].to(dtype=torch.bool)
                        if kwargs.get("attention_mask", None) is not None
                        else None
                    ),
                    condition_mask=kwargs.get("condition_mask", None),
                    padding_mask=kwargs.get("padding_mask", None),
                )

        # Build hidden_states tuple from captured intermediate features
        # Also reshape from [B, C, T, H, W] to [B, N_tokens, hidden_dim]
        extracted = []
        if visual_token_features is not None:
            # ActionDiT deliberately keeps the dense visual interface.  The
            # auxiliary depth tokens are never substituted into hidden_states.
            extracted.append(visual_token_features)
        else:
            for feat in self._intermediate_features:
                if feat.dim() == 5:
                    # [B, C, T, H, W] -> [B, T*H*W, C]
                    B, C, T, H, W = feat.shape
                    feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
                extracted.append(feat)

        # If no hooks fired (shouldn't happen), use transformer output
        if not extracted:
            out = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            if out.dim() == 5:
                B, C, T, H, W = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(out)

        # Build compatible output object
        class _WMOutput:
            def __init__(
                self,
                hidden_states_tuple,
                sample=None,
                depth_token_features=None,
                depth_latents=None,
                loss=None,
            ):
                self.hidden_states = hidden_states_tuple
                self.depth_token_features = depth_token_features
                self.depth_latents = depth_latents
                # Compatibility only: these auxiliary tokens are not action
                # conditioning features.
                self.action_token_features = None
                # Diffusers calls this DiT prediction ``sample``.  Cosmos uses
                # it as the clean-latent estimate in the official pipeline;
                # expose it so a diagnostic can decode it without changing
                # the action path.
                self.sample = sample
                self.velocity = sample
                self.loss = loss

        sample = dit_output.sample if hasattr(dit_output, "sample") else dit_output
        return _WMOutput(
            hidden_states_tuple=tuple(extracted),
            sample=sample,
            depth_token_features=depth_token_features,
            depth_latents=depth_latents,
        )

    def encode_depth_target(self, normalized_depth_video: torch.Tensor) -> torch.Tensor:
        """Encode normalized RGB-repeated depth with the frozen Cosmos VAE."""
        return self.depth_decoder.encode_target(normalized_depth_video)

    def decode_depth(self, depth_latents: torch.Tensor) -> torch.Tensor:
        """Differentiably decode a predicted depth latent video."""
        return self.depth_decoder(depth_latents)

    @torch.inference_mode()
    def decode_depth_probe(self, clean_latents: torch.Tensor) -> torch.Tensor:
        """Decode a Cosmos DiT output for zero-shot depth visualization.

        This is intentionally separate from ``forward`` and the action loss:
        Cosmos has no native depth supervision.  The caller should convert the
        decoded VAE video to a single-channel visualization and report it as a
        qualitative probe, not metric depth.
        """
        return self.decode_depth(clean_latents)

    def generate(self, **kwargs):
        """Video generation (for world-model imagination / planning).

        This builds the full Cosmos2VideoToWorldPipeline on-the-fly.
        Not used during standard VLA training, but useful for visualization
        and planning-based approaches.
        """
        from diffusers import Cosmos2VideoToWorldPipeline

        pipe = Cosmos2VideoToWorldPipeline(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            safety_checker=None,
        )
        return pipe(**kwargs)
