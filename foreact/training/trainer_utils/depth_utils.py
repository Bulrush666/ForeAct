"""Masked depth-supervision losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_video(value: torch.Tensor, size: tuple[int, int, int], mode: str):
    if tuple(value.shape[-3:]) == size:
        return value
    kwargs = {"size": size, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(value.float(), **kwargs)


def depth_frame_mask_to_latent_mask(
    frame_mask: torch.Tensor, latent_frames: int
) -> torch.Tensor:
    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must be [B,T], got {tuple(frame_mask.shape)}")
    output_frames = frame_mask.shape[1]
    if latent_frames == 1:
        return frame_mask[:, :1]
    if (output_frames - 1) % (latent_frames - 1):
        raise ValueError(
            f"Cannot map {output_frames} decoded frames to {latent_frames} latents"
        )
    stride = (output_frames - 1) // (latent_frames - 1)
    masks = [frame_mask[:, 0]]
    for index in range(1, latent_frames):
        start = 1 + (index - 1) * stride
        stop = 1 + index * stride
        masks.append(frame_mask[:, start:stop].all(dim=1))
    return torch.stack(masks, dim=1)


def depth_latent_loss(prediction, target, frame_mask=None):
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            f"Depth latent shapes must match: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    target = target.detach()
    valid = torch.isfinite(target)
    if frame_mask is not None:
        latent_mask = depth_frame_mask_to_latent_mask(
            frame_mask.bool(), prediction.shape[2]
        )
        valid &= latent_mask[:, None, :, None, None]
    if not valid.any():
        return prediction.sum() * 0
    return (prediction.float() - target.float()).square()[valid].mean()


def depth_pixel_loss(prediction, target, valid_mask):
    size = tuple(prediction.shape[-3:])
    target = _resize_video(target, size, "trilinear")
    valid = _resize_video(valid_mask.float(), size, "nearest").bool()
    valid &= torch.isfinite(target)
    if not valid.any():
        return prediction.sum() * 0
    return F.smooth_l1_loss(
        prediction.float()[valid], target.float()[valid], beta=0.1
    )


def depth_gradient_loss(prediction, target, valid_mask):
    size = tuple(prediction.shape[-3:])
    target = _resize_video(target, size, "trilinear").float()
    valid = _resize_video(valid_mask.float(), size, "nearest").bool()
    prediction = prediction.float()
    losses = []
    horizontal = valid[..., :, 1:] & valid[..., :, :-1]
    if horizontal.any():
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        losses.append((pred_dx - target_dx).abs()[horizontal].mean())
    vertical = valid[..., 1:, :] & valid[..., :-1, :]
    if vertical.any():
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        losses.append((pred_dy - target_dy).abs()[vertical].mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0
