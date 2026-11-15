import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


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
