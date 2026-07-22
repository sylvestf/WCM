from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def image_to_pil(value: Any) -> Image.Image:
    """Convert a raw LeRobot image cell to a display-ready RGB PIL image."""

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC/CHW image, got shape {array.shape}.")

    channels_first = array.shape[0] in (1, 3, 4)
    channels_last = array.shape[-1] in (1, 3, 4)
    if channels_first and channels_last:
        raise ValueError(f"Cannot infer HWC versus CHW layout for image shape {array.shape}.")
    if channels_first:
        array = np.moveaxis(array, 0, -1)
    elif not channels_last:
        raise ValueError(f"Image must have 1, 3, or 4 channels, got shape {array.shape}.")

    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinity.")
    if np.issubdtype(array.dtype, np.floating):
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < -1e-6 or maximum > 255.0 + 1e-6:
            raise ValueError(f"Unsupported float image range [{minimum}, {maximum}].")
        if maximum <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.integer):
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < 0 or maximum > 255:
            raise ValueError(f"Unsupported integer image range [{minimum}, {maximum}].")
        array = array.astype(np.uint8, copy=False)
    else:
        raise TypeError(f"Unsupported image dtype {array.dtype}.")

    channels = array.shape[-1]
    if channels == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")
    if channels == 4:
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    return Image.fromarray(array, mode="RGB")
