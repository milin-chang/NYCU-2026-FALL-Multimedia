import csv
from contextlib import nullcontext
from statistics import mean

import numpy as np
import torch
from torch.utils.data import Dataset

from prepare import BASICSR_REF, NIQE_ASSETS
from sr import SEED, read_rgb


PAIRED_METRICS = ('psnr', 'ssim', 'lpips', 'dists')
KID_SUBSETS = 100


class InceptionImages(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return rgb_tensor(np.array(read_rgb(self.paths[index])))


def rgb_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def inception_scores(sr_paths, hr_paths, device):
    subset_size = min(100, len(sr_paths))
    settings = {'implementation': 'torch-fidelity==0.3.0', 'extractor': 'inception-v3-compat',
                'feature_layer': 2048, 'kid_subsets': KID_SUBSETS, 'kid_subset_size': subset_size,
                'seed': SEED}
    if subset_size < 2:
        return {'fid': None, 'kid': None, 'kid_std': None}, settings

    import torch_fidelity as fidelity

    use_cuda = device.type == 'cuda'
    with torch.cuda.device(device) if use_cuda else nullcontext():
        scores = fidelity.calculate_metrics(
            input1=InceptionImages(sr_paths), input2=InceptionImages(hr_paths),
            cuda=use_cuda, batch_size=8, isc=False, fid=True, kid=True, cache=False, save_cpu_ram=True,
            feature_extractor='inception-v3-compat', feature_layer_fid='2048', feature_layer_kid='2048',
            kid_subsets=KID_SUBSETS, kid_subset_size=subset_size, rng_seed=SEED, verbose=False,
        )
    return {'fid': float(scores[fidelity.KEY_METRIC_FID]),
            'kid': float(scores[fidelity.KEY_METRIC_KID_MEAN]),
            'kid_std': float(scores[fidelity.KEY_METRIC_KID_STD])}, settings


def extend_scores(out, hr_dir, names, device):
    from pytorch_msssim import MS_SSIM

    if not (NIQE_ASSETS / 'niqe_pris_params.npz').is_file():
        raise FileNotFoundError('Run python prepare.py assets to install the NIQE files.')
    from assets.upstream.basicsr_niqe.niqe import calculate_niqe

    with (out / 'per_image.csv').open(newline='', encoding='utf-8') as stream:
        paired = list(csv.DictReader(stream))
    if len(paired) != len(names) or {row['image'] for row in paired} != set(names):
        raise ValueError('Existing per_image.csv does not match the selected manifest split.')

    ms_ssim = MS_SSIM(data_range=255, channel=3).to(device)
    rows = []
    for row in paired:
        name = row['image']
        sr = np.array(read_rgb(out / 'sr' / name))
        hr = np.array(read_rgb(hr_dir / name))
        if sr.shape != hr.shape:
            raise ValueError(f'SR/HR shape mismatch for {name}.')
        pred, target = (rgb_tensor(image).unsqueeze(0).to(device, dtype=torch.float32) for image in (sr, hr))
        with torch.inference_mode():
            structural = ms_ssim(pred, target).item()
        naturalness = calculate_niqe(sr[:, :, ::-1], crop_border=0, input_order='HWC', convert_to='y')
        rows.append({'image': name, **{key: float(row[key]) for key in PAIRED_METRICS},
                     'ms_ssim': structural, 'niqe': naturalness})

    ordered = sorted(names)
    distribution, settings = inception_scores([out / 'sr' / name for name in ordered],
                                              [hr_dir / name for name in ordered], device)
    averages = {key: mean(row[key] for row in rows) for key in (*PAIRED_METRICS, 'ms_ssim', 'niqe')}
    protocol = {'ms_ssim': {'implementation': 'pytorch-msssim==1.0.0', 'color': 'RGB', 'scales': 5},
                'niqe': {'implementation': f'BasicSR@{BASICSR_REF}', 'color': 'YCbCr Y', 'crop_border': 0},
                'fid_kid': settings}
    return rows, {**averages, **distribution, 'metric_protocol': protocol}
