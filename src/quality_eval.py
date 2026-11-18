"""Eight full-reference RGB quality metrics used only for validation and testing."""

import math

import numpy as np
import torch


METRIC_DIRECTIONS = {
    "psnr": "higher",
    "ssim": "higher",
    "lpips": "lower",
    "dists": "lower",
    "ms_ssim": "higher",
    "fsim": "higher",
    "vifp": "higher",
    "uqi": "higher",
}


class MetricSuite:
    def __init__(self, device):
        import DISTS_pytorch
        import lpips

        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        self.dists = DISTS_pytorch.DISTS(load_weights=True).eval().to(device)
        self.lpips.requires_grad_(False)
        self.dists.requires_grad_(False)

    @torch.inference_mode()
    def score(self, sr_rgb8: np.ndarray, hr_rgb8: np.ndarray) -> dict[str, float]:
        import piq
        from sewar.full_ref import uqi

        if sr_rgb8.shape != hr_rgb8.shape or sr_rgb8.ndim != 3 or sr_rgb8.shape[2] != 3:
            raise ValueError("SR and HR must be same-sized RGB images.")
        sr = torch.from_numpy(sr_rgb8.copy()).permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=torch.float32) / 255.0
        hr = torch.from_numpy(hr_rgb8.copy()).permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=torch.float32) / 255.0
        values = {
            "psnr": piq.psnr(sr, hr, data_range=1.0).item(),
            "ssim": piq.ssim(sr, hr, data_range=1.0).item(),
            "lpips": self.lpips(sr * 2 - 1, hr * 2 - 1).item(),
            "dists": self.dists(sr, hr).item(),
            "ms_ssim": piq.multi_scale_ssim(sr, hr, data_range=1.0).item(),
            "fsim": piq.fsim(sr, hr, data_range=1.0, chromatic=False).item(),
            "vifp": piq.vif_p(sr, hr, data_range=1.0).item(),
            "uqi": float(uqi(hr_rgb8, sr_rgb8)),
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite image quality score: {values}")
        return values
