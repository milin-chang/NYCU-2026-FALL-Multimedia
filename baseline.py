import argparse
import copy
import csv
import json
import platform
import random
import time
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset, Sampler

from prepare import DEFAULT_DATA, sha256
from sr import (KAIR_REF, SEED, SWINIR_REF, WEIGHT, WINDOW, build_generator, degrade, image_tensor,
                read_rgb, rgb8, upscale, upstream)


class TrainImages(Dataset):
    def __init__(self, root, rows):
        self.paths = [root / 'train' / 'hr' / row['name'] for row in rows]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        index, position = key
        return degrade(read_rgb(self.paths[index]), SEED + position, training=True)


class TrainOrder(Sampler):
    def __init__(self, size, batch, start, end):
        self.size = size
        self.batch = batch
        self.start = start
        self.end = end

    def __iter__(self):
        usable = self.size // self.batch * self.batch
        epoch = -1
        for position in range(self.start * self.batch, self.end * self.batch):
            next_epoch, offset = divmod(position, usable)
            if epoch != next_epoch:
                epoch = next_epoch
                order = list(range(self.size))
                random.Random(SEED + epoch).shuffle(order)
            yield order[offset], position

    def __len__(self):
        return (self.end - self.start) * self.batch


def seed_model():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def manifest(data):
    return json.loads((data / 'manifest.json').read_text(encoding='utf-8'))['splits']


def perceptual_metrics(device, full=False):
    import lpips

    metrics = {'lpips': lpips.LPIPS(net='alex', verbose=False).eval().to(device)}
    if full:
        import DISTS_pytorch

        dists = DISTS_pytorch.DISTS(load_weights=False)
        weights = Path(DISTS_pytorch.__file__).with_name('weights.pt')
        state = torch.load(weights, map_location='cpu', weights_only=True)
        with torch.no_grad():
            dists.alpha.copy_(state['alpha'])
            dists.beta.copy_(state['beta'])
        metrics['dists'] = dists.eval().to(device)
    return metrics


@torch.inference_mode()
def evaluate(model, data, split, rows, metrics, device, tile, output=None):
    scores = []
    if output:
        (output / 'sr').mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row['name']
        lr = image_tensor(read_rgb(data / split / 'lr' / name)).to(device)
        hr = np.asarray(read_rgb(data / split / 'hr' / name))
        sr = rgb8(upscale(model, lr, tile))
        if hr.shape != sr.shape:
            raise ValueError(f'LR/HR shape mismatch for {name}.')
        result = {
            'image': name,
            'psnr': float(peak_signal_noise_ratio(hr, sr, data_range=255)),
            'ssim': float(structural_similarity(hr, sr, channel_axis=2, data_range=255,
                                               gaussian_weights=True, use_sample_covariance=False)),
        }
        # The perceptual metrics see the same quantized RGB images as the saved PNGs.
        pred, target = [image_tensor(rgb.astype(np.float32) / 255).unsqueeze(0).to(device)
                        for rgb in (sr, hr)]
        result['lpips'] = metrics['lpips'](pred * 2 - 1, target * 2 - 1).item()
        if 'dists' in metrics:
            result['dists'] = metrics['dists'](pred, target).item()
        if output:
            Image.fromarray(sr).save(output / 'sr' / name)
        scores.append(result)
    average = {key: mean(row[key] for row in scores) for key in scores[0] if key != 'image'}
    if output:
        with (output / 'per_image.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
            writer.writeheader()
            writer.writerows(scores)
    return average


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def append_log(path, event, **values):
    line = json.dumps({'event': event, **values})
    print(line, flush=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')


class GanTraining:
    ema_decay = 0.999

    def __init__(self, args, device):
        upstream('kair')
        from models.loss import GANLoss, PerceptualLoss
        from models.select_network import define_D

        self.device_type = device.type
        self.amp = False
        if device.type == 'cuda' and not args.fp32:
            with torch.cuda.device(device):
                self.amp = torch.cuda.is_bf16_supported()
        self.model = build_generator(args.weights, args.activation_checkpoint, args.fft).to(device)
        self.ema = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.discriminator = define_D({'netD': {'net_type': 'discriminator_unet', 'in_nc': 3, 'base_nc': 64,
                                                'init_type': 'orthogonal', 'init_bn_type': 'uniform',
                                                'init_gain': 0.2}}).to(device)
        self.feature_loss = PerceptualLoss().requires_grad_(False).eval().to(device)
        self.adversarial = GANLoss('gan').to(device)
        self.opt_g = torch.optim.Adam(self.model.parameters(), lr=args.g_lr)
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.d_lr)

    def step(self, lr, hr, step):
        self.discriminator.requires_grad_(False)
        self.opt_g.zero_grad(set_to_none=True)
        with torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=self.amp):
            sr = self.model(lr)
            pixel = F.l1_loss(sr, hr)
            feature = self.feature_loss(sr, hr)
            gan = self.adversarial(self.discriminator(sr), True)
            loss_g = pixel + feature + 0.1 * gan
        if not loss_g.isfinite().item():
            raise FloatingPointError(f'Non-finite generator loss at step {step}.')
        loss_g.backward()
        self.opt_g.step()

        self.discriminator.requires_grad_(True)
        self.opt_d.zero_grad(set_to_none=True)
        with torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=self.amp):
            d_real = self.adversarial(self.discriminator(hr), True)
            d_fake = self.adversarial(self.discriminator(sr.detach()), False)
            loss_d = d_real + d_fake
        if not loss_d.isfinite().item():
            raise FloatingPointError(f'Non-finite discriminator loss at step {step}.')
        loss_d.backward()
        self.opt_d.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(self.ema.parameters()), list(self.model.parameters()), 1 - self.ema_decay)
        return {'pixel': pixel.detach(), 'feature': feature.detach(), 'gan': gan.detach(),
                'g_total': loss_g.detach(), 'd_real': d_real.detach(), 'd_fake': d_fake.detach()}


def train(args, device):
    seed_model()
    data = manifest(args.data)
    if args.batch > len(data['train']):
        raise ValueError('--batch exceeds the training set size.')
    log = args.run / 'events.jsonl'
    last = args.run / 'last.pth'
    if not args.resume and (last.exists() or log.exists()):
        raise FileExistsError('Use --resume for a previous run or choose a new --run directory.')
    training = GanTraining(args, device)
    config = {'seed': SEED, 'swinir_ref': SWINIR_REF, 'kair_ref': KAIR_REF,
              'manifest_sha256': sha256(args.data / 'manifest.json'), 'weight_sha256': sha256(args.weights),
              'batch': args.batch, 'g_lr': args.g_lr, 'd_lr': args.d_lr,
              'precision': 'bf16' if training.amp else 'fp32',
              'ema_decay': training.ema_decay, 'loss_weights': [1., 1., 0.1]}
    if hasattr(training.model.patch_unembed, 'fft_branch'):
        config['fft'] = True
    model, ema, discriminator = training.model, training.ema, training.discriminator
    opt_g, opt_d = training.opt_g, training.opt_d
    metrics = perceptual_metrics(device)
    start, best = 0, float('inf')
    if args.resume:
        state = torch.load(last, map_location='cpu', weights_only=True)
        if state['config'] != config:
            raise ValueError('Resume settings differ from last.pth.')
        for component, key in [(model, 'params'), (ema, 'params_ema'), (discriminator, 'discriminator'),
                               (opt_g, 'optimizer_g'), (opt_d, 'optimizer_d')]:
            component.load_state_dict(state[key])
        start, best = state['step'], state['best_lpips']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        del state
    if start >= args.steps:
        raise ValueError(f'--steps must exceed checkpoint step {start}.')

    append_log(log, 'resume' if args.resume else 'start', step=start, steps=args.steps,
               config=config, python=platform.python_version(), torch=torch.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               workers=args.workers, activation_checkpoint=args.activation_checkpoint, tile=args.tile)
    if start == 0:
        append_log(log, 'validation', step=0, **evaluate(ema, args.data, 'val', data['val'], metrics, device, args.tile))

    images = TrainImages(args.data, data['train'])
    order = TrainOrder(len(images), args.batch, start, args.steps)
    loader = DataLoader(images, batch_size=args.batch, sampler=order, num_workers=args.workers,
                        pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(SEED))
    began = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for step, (lr, hr) in enumerate(loader, start + 1):
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
        losses = training.step(lr, hr, step)

        if step == start + 1 or step % args.log_every == 0 or step == args.steps:
            append_log(log, 'train', step=step, **{name: loss.item() for name, loss in losses.items()},
                       elapsed_s=round(time.monotonic() - began, 2),
                       peak_vram_mib=round(torch.cuda.max_memory_allocated(device) / 1048576, 1)
                       if device.type == 'cuda' else None)
        if step % args.val_every == 0 or step == args.steps:
            scores = evaluate(ema, args.data, 'val', data['val'], metrics, device, args.tile)
            if scores['lpips'] < best:
                best = scores['lpips']
                save_checkpoint(args.run / 'best.pth', {'params_ema': ema.state_dict(), 'step': step, 'val': scores})
            save_checkpoint(last, {
                'step': step, 'config': config, 'best_lpips': best, 'params': model.state_dict(),
                'params_ema': ema.state_dict(), 'discriminator': discriminator.state_dict(),
                'optimizer_g': opt_g.state_dict(), 'optimizer_d': opt_d.state_dict(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            })
            append_log(log, 'validation', step=step, best_lpips=best, **scores)


def evaluate_split(args, device):
    from quality_metrics import extend_scores

    rows = manifest(args.data)[args.split]
    names = [row['name'] for row in rows]
    metadata = {'split': args.split, 'images': len(rows), 'tile': args.tile,
                'weights_sha256': sha256(args.weights), 'manifest_sha256': sha256(args.data / 'manifest.json')}
    args.out.mkdir(parents=True, exist_ok=True)
    with FileLock(args.out / '.evaluate.lock', timeout=0), TemporaryDirectory(dir=args.out) as temp:
        stage = Path(temp)
        if args.reuse_sr:
            prior = json.loads((args.out / 'summary.json').read_text(encoding='utf-8'))
            if any(prior.get(key) != value for key, value in metadata.items()):
                raise ValueError('Existing report metadata differs from the requested weights, split, or tile.')
            source = args.out
        else:
            model = build_generator(args.weights).eval().to(device)
            evaluate(model, args.data, args.split, rows, perceptual_metrics(device, full=True),
                     device, args.tile, stage)
            del model
            source = stage

        per_image, scores = extend_scores(source, args.data / args.split / 'hr', names, device)
        report = json.dumps({**metadata, **scores}, indent=2) + '\n'
        (stage / 'summary.json').write_text(report, encoding='utf-8')
        with (stage / 'per_image.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
            writer.writeheader()
            writer.writerows(per_image)
        if not args.reuse_sr:
            (args.out / 'sr').mkdir(exist_ok=True)
            for name in names:
                (stage / 'sr' / name).replace(args.out / 'sr' / name)
        (stage / 'per_image.csv').replace(args.out / 'per_image.csv')
        (stage / 'summary.json').replace(args.out / 'summary.json')
    print(report, end='')


def inference(args, device):
    model = build_generator(args.weights).eval().to(device)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
    images = [args.input] if args.input.is_file() else sorted(
        path for path in args.input.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    if not images:
        raise ValueError(f'No input images under {args.input}.')
    if len({path.stem.casefold() for path in images}) != len(images):
        raise ValueError('Input file stems must be unique when saving PNGs.')
    if any(path.resolve() == (args.out / f'{path.stem}.png').resolve() for path in images):
        raise ValueError('Choose an output directory that preserves the original inputs.')
    args.out.mkdir(parents=True, exist_ok=True)
    for path in images:
        pred = upscale(model, image_tensor(read_rgb(path)).to(device), args.tile)
        target = args.out / f'{path.stem}.png'
        Image.fromarray(rgb8(pred)).save(target)
        print(target)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--device', default='cuda', help='cuda, cuda:0, or cpu')
    common.add_argument('--tile', type=int, default=128, help='LR tile size, 0 for whole-image inference')
    common.add_argument('--weights', type=Path, default=WEIGHT)
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train', parents=[common])
    fit.add_argument('--data', type=Path, default=DEFAULT_DATA)
    fit.add_argument('--run', type=Path, default=Path('runs/baseline'))
    fit.add_argument('--steps', type=int, default=100_000)
    fit.add_argument('--batch', type=int, default=1)
    fit.add_argument('--workers', type=int, default=0)
    fit.add_argument('--g-lr', type=float, default=1e-5)
    fit.add_argument('--d-lr', type=float, default=1e-4)
    fit.add_argument('--log-every', type=int, default=100)
    fit.add_argument('--val-every', type=int, default=5000)
    fit.add_argument('--activation-checkpoint', action='store_true')
    fit.add_argument('--fp32', action='store_true')
    fit.add_argument('--fft', action='store_true', help='Add a zero-initialized FFT residual branch')
    fit.add_argument('--resume', action='store_true')
    infer = commands.add_parser('infer', parents=[common])
    infer.add_argument('--input', type=Path, required=True)
    infer.add_argument('--out', type=Path, required=True)
    scoring = commands.add_parser('evaluate', parents=[common])
    scoring.add_argument('--data', type=Path, default=DEFAULT_DATA)
    scoring.add_argument('--split', choices=('val', 'test'), default='test')
    scoring.add_argument('--out', type=Path, required=True)
    scoring.add_argument('--reuse-sr', action='store_true', help='Extend an existing report using its saved SR PNGs')
    args = parser.parse_args()
    if args.tile < 0 or args.tile % WINDOW:
        parser.error('--tile must be 0 or a positive multiple of 8.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Check the NVIDIA driver and PyTorch build, or use --device cpu.')
    if args.command == 'train':
        if min(args.steps, args.batch, args.val_every, args.log_every, args.g_lr, args.d_lr) <= 0 or args.workers < 0:
            parser.error('Steps, batch, intervals and learning rates must be positive, and workers at least zero.')
        args.run.mkdir(parents=True, exist_ok=True)
        with FileLock(args.run / '.train.lock', timeout=0):
            train(args, device)
    elif args.command == 'evaluate':
        evaluate_split(args, device)
    else:
        inference(args, device)


if __name__ == '__main__':
    main()
