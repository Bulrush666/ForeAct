"""Cosmos-compatible depth target encoder and differentiable VAE decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DepthVideoDecoder(nn.Module):
    def __init__(self, vae: nn.Module, scheduler, gradient_checkpointing: bool = False):
        super().__init__()
        object.__setattr__(self, "_vae", vae)
        self.scheduler = scheduler
        self.gradient_checkpointing = bool(gradient_checkpointing)

    @property
    def vae(self) -> nn.Module:
        return self._vae

    def _latent_stats(self, reference: torch.Tensor):
        config = self.vae.config
        if getattr(config, "latents_mean", None) is None:
            return None
        mean = torch.as_tensor(
            config.latents_mean, device=reference.device, dtype=reference.dtype
        ).view(1, config.z_dim, 1, 1, 1)
        std = torch.as_tensor(
            config.latents_std, device=reference.device, dtype=reference.dtype
        ).view(1, config.z_dim, 1, 1, 1)
        return mean, std

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        stats = self._latent_stats(latents)
        if stats is None:
            return latents
        mean, std = stats
        return (latents - mean) / std * self.scheduler.config.sigma_data

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        stats = self._latent_stats(latents)
        if stats is None:
            return latents
        mean, std = stats
        return latents * std / self.scheduler.config.sigma_data + mean

    @torch.no_grad()
    def encode_target(self, normalized_depth: torch.Tensor) -> torch.Tensor:
        if normalized_depth.ndim != 5 or normalized_depth.shape[1] != 1:
            raise ValueError(
                f"normalized depth must be [B,1,T,H,W], got {tuple(normalized_depth.shape)}"
            )
        depth_rgb = normalized_depth.expand(-1, 3, -1, -1, -1).contiguous()
        posterior = self.vae.encode(depth_rgb.to(dtype=self.vae.dtype)).latent_dist
        latents = posterior.mode() if hasattr(posterior, "mode") else posterior.mean
        return self.normalize_latents(latents)

    def forward(self, depth_latents: torch.Tensor) -> torch.Tensor:
        latents = self.denormalize_latents(depth_latents)

        def decode(value: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(
                value.to(dtype=self.vae.dtype), return_dict=False
            )[0]

        device_type = latents.device.type
        with torch.amp.autocast(
            device_type=device_type, dtype=self.vae.dtype,
            enabled=device_type == "cuda",
        ):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                decoded = checkpoint(decode, latents, use_reentrant=False)
            else:
                decoded = decode(latents)
        if decoded.ndim != 5 or decoded.shape[1] != 3:
            raise ValueError(f"VAE decoder returned {tuple(decoded.shape)}")
        return decoded.float()
