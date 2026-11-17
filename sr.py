import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn


SCALE = 4
WINDOW = 8
LR_PATCH = 64
HR_SIZE = 512
SEED = 42
WEIGHT = Path('assets/weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
SWINIR_REF = '6545850fbf8df298df73d81f3e8cba638787c8bd'
KAIR_REF = 'fc1732f4a4514e42ce15e5b3a1e18c828af47a1e'


def upstream(name):
    path = Path(__file__).resolve().parent / 'assets' / 'upstream' / name
    if not path.is_dir():
        raise FileNotFoundError('Run python prepare.py assets first.')
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def generator(checkpoint_layers=False):
    upstream('swinir')
    from network_swinir import SwinIR

    return SwinIR(upscale=SCALE, in_chans=3, img_size=LR_PATCH, window_size=WINDOW,
                  img_range=1., depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                  mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv',
                  use_checkpoint=checkpoint_layers)


def load_generator(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    return state['params_ema']


class FFTBranch(nn.Module):
    def __init__(self, channels=180, width=32):
        super().__init__()
        self.reduce = nn.Conv2d(channels, width, 1)
        self.mix = nn.Conv2d(2 * width, 2 * width, 1)
        self.project = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features):
        reduced = self.reduce(features)
        with torch.autocast(features.device.type, enabled=False):
            spectrum = torch.fft.rfft2(reduced.float(), norm='ortho')
            mixed = F.leaky_relu(self.mix(torch.cat((spectrum.real, spectrum.imag), dim=1)), 0.1)
            real, imag = mixed.chunk(2, dim=1)
            restored = torch.fft.irfft2(torch.complex(real, imag), s=reduced.shape[-2:], norm='ortho')
        return self.project(restored.to(reduced.dtype)).to(features.dtype)


def fft_residual(module, _inputs, features):
    return features + module.fft_branch(features)


def haar_dwt2(features):
    height, width = features.shape[-2:]
    if height % 2 or width % 2:
        features = F.pad(features, (0, width % 2, 0, height % 2), mode='replicate')
    tl, tr, bl, br = F.pixel_unshuffle(features, 2).unflatten(1, (-1, 4)).unbind(2)
    left_sum, right_sum = tl + bl, tr + br
    left_diff, right_diff = tl - bl, tr - br
    return torch.cat((left_sum + right_sum, left_diff + right_diff,
                      left_sum - right_sum, left_diff - right_diff), dim=1) * 0.5


def haar_idwt2(bands, size):
    ll, lh, hl, hh = bands.chunk(4, dim=1)
    left_sum, right_sum = ll + hl, ll - hl
    left_diff, right_diff = lh + hh, lh - hh
    cells = torch.stack((left_sum + left_diff, right_sum + right_diff,
                         left_sum - left_diff, right_sum - right_diff), dim=2) * 0.5
    height, width = size
    return F.pixel_shuffle(cells.flatten(1, 2), 2)[..., :height, :width]


class DWTBranch(nn.Module):
    def __init__(self, channels=180, width=32):
        super().__init__()
        self.reduce = nn.Conv2d(channels, width, 1)
        self.mix = nn.Conv2d(4 * width, 4 * width, 1, groups=4)
        self.project = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features):
        reduced = self.reduce(features)
        with torch.autocast(features.device.type, enabled=False):
            bands = haar_dwt2(reduced.float())
            mixed = F.leaky_relu(self.mix(bands), 0.1)
            restored = haar_idwt2(mixed, reduced.shape[-2:])
        return self.project(restored.to(reduced.dtype)).to(features.dtype)


def dwt_residual(module, _inputs, features):
    return features + module.dwt_branch(features)


def build_generator(weights, checkpoint_layers=False, fft=False, dwt=False):
    if fft and dwt:
        raise ValueError('Select either FFT or DWT.')
    model = generator(checkpoint_layers)
    state = load_generator(weights)
    variants = {'fft': (FFTBranch, fft_residual), 'dwt': (DWTBranch, dwt_residual)}
    saved = [name for name in variants if any(key.startswith(f'patch_unembed.{name}_branch.') for key in state)]
    requested = 'fft' if fft else 'dwt' if dwt else None
    if len(saved) > 1 or (saved and requested and requested != saved[0]):
        raise ValueError('Checkpoint frequency branch does not match the requested architecture.')
    if not saved:
        model.load_state_dict(state, strict=True)
    variant = saved[0] if saved else requested
    if variant:
        branch_type, hook = variants[variant]
        # Keep the original initialization and the subsequent discriminator RNG sequence.
        with torch.random.fork_rng(devices=[]):
            branch = branch_type()
        model.patch_unembed.add_module(f'{variant}_branch', branch)
        model.patch_unembed.register_forward_hook(hook)
    if saved:
        model.load_state_dict(state, strict=True)
    return model


def read_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def image_tensor(image):
    pixels = np.asarray(image, dtype=np.float32) / 255 if isinstance(image, Image.Image) else image
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32))


def rgb8(tensor):
    return tensor.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def degrade(image, seed, training):
    upstream('kair')
    from utils.utils_blindsr import degradation_bsrgan

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    if training:
        patch = LR_PATCH * SCALE
        x = random.randrange(image.width - patch + 1)
        y = random.randrange(image.height - patch + 1)
        image = image.crop((x, y, x + patch, y + patch))
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
    lr, hr = degradation_bsrgan(np.asarray(image, dtype=np.float32) / 255,
                                sf=SCALE, lq_patchsize=image.width // SCALE)
    return image_tensor(lr), image_tensor(hr)


@torch.inference_mode()
def upscale(model, image, tile):
    h, w = image.shape[-2:]
    pad_h, pad_w = (-h) % WINDOW, (-w) % WINDOW
    mode = 'reflect' if pad_h < h and pad_w < w else 'replicate'
    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode)
    hp, wp = padded.shape[-2:]
    if not tile or max(hp, wp) <= tile:
        return model(padded)[0, :, :h * SCALE, :w * SCALE]

    output = image.new_empty((3, hp * SCALE, wp * SCALE))
    halo = 16
    for y in range(0, hp, tile):
        for x in range(0, wp, tile):
            y1, x1 = min(y + tile, hp), min(x + tile, wp)
            top, left = max(0, y - halo), max(0, x - halo)
            patch = padded[:, :, top:y1 + halo, left:x1 + halo]
            pred = model(patch)[0]
            output[:, y * SCALE:y1 * SCALE, x * SCALE:x1 * SCALE] = pred[
                :, (y - top) * SCALE:(y1 - top) * SCALE, (x - left) * SCALE:(x1 - left) * SCALE]
    return output[:, :h * SCALE, :w * SCALE]
