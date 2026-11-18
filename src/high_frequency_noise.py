"""Reproducible high-pass Gaussian noise for synthetic FFHQ LR images."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def high_pass_filter(height, width, cutoff):
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius_sq = fx * fx + fy * fy
    return 1 - np.exp(-radius_sq / (2 * cutoff * cutoff))


def high_frequency_noise(shape, seed, cutoff=0.08):
    """Return zero-mean, unit-std high-frequency noise for an HWC image."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(shape)
    filt = high_pass_filter(shape[0], shape[1], cutoff)
    result = np.fft.ifft2(np.fft.fft2(white, axes=(0, 1)) * filt[..., None],
                           axes=(0, 1)).real
    result -= result.mean(axis=(0, 1), keepdims=True)
    result /= result.std(axis=(0, 1), keepdims=True)
    return result.astype(np.float32)


def add_high_frequency_noise(rgb8, sigma_8bit, seed, cutoff=0.08):
    """Add noise after bicubic downsampling, then quantize to RGB uint8."""
    if rgb8.dtype != np.uint8 or rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError("Expected an HWC RGB uint8 image.")
    noise = high_frequency_noise(rgb8.shape, seed=seed, cutoff=cutoff)
    return np.clip(np.rint(rgb8.astype(np.float32) + sigma_8bit * noise),
                   0, 255).astype(np.uint8)
