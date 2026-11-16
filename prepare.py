import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from sr import HR_SIZE, KAIR_REF, SEED, SWINIR_REF, WEIGHT, degrade, read_rgb, rgb8


KAGGLE = 'arnaud58/flickrfaceshq-dataset-ffhq/versions/1'
RAW = 'https://raw.githubusercontent.com'
WEIGHT_SHA256 = 'b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6'
DEFAULT_DATA = Path('data/ffhq')
BASICSR_REF = '8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a'
NIQE_ASSETS = Path('assets/upstream/basicsr_niqe')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, path, expected=None):
    if path.is_file() and (expected is None or sha256(path) == expected):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    with urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
        shutil.copyfileobj(response, stream)
    if expected and sha256(temporary) != expected:
        raise ValueError(f'Unexpected SHA-256 for {url}')
    temporary.replace(path)


def patch_asset(path, replacements):
    contents = path.read_bytes()
    patched = contents
    for source, replacement in replacements:
        patched = patched.replace(source, replacement)
    if patched != contents:
        temporary = path.with_suffix('.part')
        temporary.write_bytes(patched)
        temporary.replace(path)


def assets():
    release = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0'
    download(f'{release}/{WEIGHT.name}', WEIGHT, WEIGHT_SHA256)
    files = {
        'JingyunLiang/SwinIR': (SWINIR_REF, 'swinir', ['models/network_swinir.py', 'LICENSE']),
        'cszn/KAIR': (KAIR_REF, 'kair', ['models/basicblock.py', 'models/network_discriminator.py',
                                       'models/select_network.py', 'models/loss.py',
                                       'utils/utils_blindsr.py', 'utils/utils_image.py', 'LICENSE']),
    }
    for repo, (ref, name, paths) in files.items():
        directory = Path('assets/upstream') / name
        for path in paths:
            target = Path(path).name if name == 'swinir' else path
            download(f'{RAW}/{repo}/{ref}/{path}', directory / target)
    # KAIR uses a SciPy alias removed from current versions. NumPy is the original implementation.
    blindsr = Path('assets/upstream/kair/utils/utils_blindsr.py')
    patch_asset(blindsr, [(b'scipy.finfo(', b'np.finfo(')])

    # Load the NIQE modules in isolation so they do not pull in BasicSR training extensions.
    basicsr = f'{RAW}/XPixelGroup/BasicSR/{BASICSR_REF}'
    for name in ('LICENSE.txt', 'basicsr/metrics/niqe.py', 'basicsr/metrics/niqe_pris_params.npz',
                 'basicsr/metrics/metric_util.py', 'basicsr/utils/matlab_functions.py',
                 'basicsr/utils/color_util.py', 'basicsr/utils/registry.py'):
        download(f'{basicsr}/{name}', NIQE_ASSETS / Path(name).name)
    patch_asset(NIQE_ASSETS / 'niqe.py', [
        (b'from basicsr.metrics.metric_util import', b'from .metric_util import'),
        (b'from basicsr.utils.matlab_functions import', b'from .matlab_functions import'),
        (b'from basicsr.utils.registry import', b'from .registry import'),
    ])
    patch_asset(NIQE_ASSETS / 'metric_util.py', [(b'from basicsr.utils import', b'from .color_util import')])
    print(f'Assets ready: {WEIGHT} ({WEIGHT_SHA256})')


def dataset(args):
    if args.source is None:
        os.environ.setdefault('KAGGLEHUB_CACHE', str(Path('assets/kaggle').resolve()))
        import kagglehub

        source = Path(kagglehub.dataset_download(KAGGLE))
        source_id = KAGGLE
    else:
        source = args.source
        source_id = args.source_id
    counts = dict(zip(('train', 'val', 'test'), args.counts, strict=True))
    files = sorted(p.relative_to(source).as_posix() for p in source.rglob('*')
                   if p.is_file() and p.suffix.lower() == '.png')
    if len(files) < sum(counts.values()):
        raise ValueError(f'Found {len(files)} PNGs under {source}, need {sum(counts.values())}.')
    if len({Path(name).stem for name in files}) != len(files):
        raise ValueError('Duplicate PNG stems. Point --source at one FFHQ image tree.')
    random.Random(SEED).shuffle(files)
    manifest = args.out / 'manifest.json'
    if manifest.exists():
        raise FileExistsError(f'{manifest} exists. Use this split or choose a new --out.')
    temporary = manifest.with_suffix('.part')
    splits = {}
    offset = 0
    for split, count in counts.items():
        hr_dir = args.out / split / 'hr'
        lr_dir = args.out / split / 'lr'
        hr_dir.mkdir(parents=True, exist_ok=True)
        if split != 'train':
            lr_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, name in enumerate(files[offset:offset + count], offset):
            path = source / name
            image = read_rgb(path)
            if image.width != image.height or image.width < HR_SIZE:
                raise ValueError(f'Expected a square FFHQ PNG at least {HR_SIZE}px: {path}')
            image = image.resize((HR_SIZE, HR_SIZE), Image.Resampling.LANCZOS)
            output_name = f'{Path(name).stem}.png'
            hr_path = hr_dir / output_name
            image.save(hr_path)
            row = {'name': output_name, 'source': name, 'hr_sha256': sha256(hr_path)}
            if split != 'train':
                seed = SEED + index
                lr, _ = degrade(image, seed, training=False)
                lr_path = lr_dir / output_name
                Image.fromarray(rgb8(lr)).save(lr_path)
                row.update(lr_seed=seed, lr_sha256=sha256(lr_path))
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == count:
                print(f'{split}: {len(rows)}/{count}', flush=True)
        splits[split] = rows
        offset += count
    result = {'source': source_id, 'candidate_count': len(files), 'seed': SEED,
              'hr_size': HR_SIZE, 'degradation_ref': KAIR_REF, 'splits': splits}
    temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest)
    print(f'{manifest} ({sha256(manifest)})')


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('assets')
    data = commands.add_parser('dataset')
    data.add_argument('--out', type=Path, default=DEFAULT_DATA)
    data.add_argument('--source', type=Path, help='Existing extracted FFHQ image tree')
    data.add_argument('--source-id', default='local-ffhq')
    data.add_argument('--counts', nargs=3, metavar=('TRAIN', 'VAL', 'TEST'), type=int,
                      default=(1000, 125, 125))
    args = parser.parse_args()
    if args.command == 'assets':
        assets()
    else:
        if min(args.counts) <= 0:
            parser.error('All split counts must be positive.')
        dataset(args)


if __name__ == '__main__':
    main()
