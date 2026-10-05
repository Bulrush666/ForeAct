"""Shared deterministic image preprocessing for training and deployment.

The model configuration uses height/width order (``*_hw``), while Pillow uses
width/height.  Keeping the conversion here avoids the historic train/eval
resize mismatch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image


def preprocess_observation_image(
    image,
    *,
    crop_xyxy: Sequence[int] | None = None,
    size_hw: Sequence[int] | None = None,
) -> Image.Image:
    """Crop and resize one observation image, returning an RGB PIL image.

    Images already at ``size_hw`` are treated as preprocessed.  This makes the
    helper idempotent when evaluation happens to receive a packed dataset
    sample instead of a raw camera frame.
    """

    if isinstance(image, Image.Image):
        output = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size and float(array.max()) <= 1.0:
                array = np.clip(array * 255.0, 0.0, 255.0)
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        output = Image.fromarray(array).convert("RGB")

    target_size = None
    if size_hw is not None:
        if len(size_hw) != 2:
            raise ValueError(f"obs_image_size_hw must contain [height, width], got {size_hw}")
        target_h, target_w = (int(size_hw[0]), int(size_hw[1]))
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"obs_image_size_hw must be positive, got {size_hw}")
        target_size = (target_w, target_h)

    if target_size is not None and output.size == target_size:
        return output

    if crop_xyxy is not None:
        if len(crop_xyxy) != 4:
            raise ValueError(f"obs_image_crop_xyxy must contain [left, top, right, bottom], got {crop_xyxy}")
        left, top, right, bottom = (int(value) for value in crop_xyxy)
        width, height = output.size
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(
                f"crop {list(crop_xyxy)} is outside input image size {(height, width)}"
            )
        output = output.crop((left, top, right, bottom))

    if target_size is not None and output.size != target_size:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        output = output.resize(target_size, resampling)
    return output


def preprocess_observation_images(images, *, crop_xyxy=None, size_hw=None):
    """Apply :func:`preprocess_observation_image` to a nested image container."""

    if isinstance(images, (list, tuple)):
        converted = [
            preprocess_observation_images(item, crop_xyxy=crop_xyxy, size_hw=size_hw)
            for item in images
        ]
        return type(images)(converted)
    return preprocess_observation_image(images, crop_xyxy=crop_xyxy, size_hw=size_hw)
