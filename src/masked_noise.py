"""Reproducible face-region masked Gaussian noise for synthetic FFHQ LR images."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def soft_face_mask(height, width):
    """Central soft ellipse used by the earlier FFHQ masked-noise preview."""
    yy, xx = np.mgrid[:height, :width]
    radius_sq = ((xx - width * 0.50) / (width * 0.34)) ** 2
    radius_sq += ((yy - height * 0.52) / (height * 0.36)) ** 2
    return np.clip((1.10 - radius_sq) / 0.20, 0, 1).astype(np.float32)


def add_masked_noise(rgb8, sigma_8bit, seed):
    """Add independent RGB Gaussian noise only within the soft face ellipse."""
    if rgb8.dtype != np.uint8 or rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError("Expected an HWC RGB uint8 image.")
    mask = soft_face_mask(*rgb8.shape[:2])[..., None]
    noise = np.random.default_rng(seed).standard_normal(rgb8.shape)
    return np.clip(np.rint(rgb8.astype(np.float32) + sigma_8bit * noise * mask),
                   0, 255).astype(np.uint8)
