import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from baseline import GanTraining, perceptual_metrics, seed_model
from sr import HR_SIZE, LR_PATCH, SCALE, WEIGHT, upscale


MIB = 1024 ** 2


def trial(args, device):
    seed_model()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    info = {'batch': args.trial, 'gpu': torch.cuda.get_device_name(device),
            'total_mib': round(total / MIB, 1), 'free_at_start_mib': round(free / MIB, 1),
            'activation_checkpoint': args.activation_checkpoint, 'fft': args.fft}
    try:
        training = GanTraining(args, device)
        info['precision'] = 'bf16' if training.amp else 'fp32'
        metrics = perceptual_metrics(device)

        lr = torch.rand(args.trial, 3, LR_PATCH, LR_PATCH, device=device)
        hr = torch.rand(args.trial, 3, LR_PATCH * SCALE, LR_PATCH * SCALE, device=device)
        training.step(lr, hr, 1)
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        for step in range(2, args.steps + 2):
            training.step(lr, hr, step)
        torch.cuda.synchronize(device)
        seconds = (time.perf_counter() - began) / args.steps
        del lr, hr

        # Exercise the validation image size with Adam state and training gradients still resident.
        with torch.inference_mode():
            small = torch.rand(3, HR_SIZE // SCALE, HR_SIZE // SCALE, device=device)
            pred = upscale(training.ema, small, tile=HR_SIZE // SCALE).unsqueeze(0).clamp_(0, 1)
            metrics['lpips'](pred * 2 - 1, torch.rand_like(pred) * 2 - 1)
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        return {**info, 'status': 'oom'}

    return {**info, 'status': 'ok', 'timed_steps': args.steps,
            'seconds_per_step': round(seconds, 4), 'images_per_sec': round(args.trial / seconds, 2),
            'peak_allocated_mib': round(torch.cuda.max_memory_allocated(device) / MIB, 1),
            'peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / MIB, 1)}


def run_trial(args, batch):
    flags = ['--device', args.device, '--steps', str(args.steps), '--weights', str(args.weights),
             '--trial', str(batch)]
    for enabled, flag in [(args.fp32, '--fp32'), (args.activation_checkpoint, '--activation-checkpoint'),
                          (args.fft, '--fft')]:
        if enabled:
            flags.append(flag)
    cmd = [sys.executable, str(Path(__file__).resolve()), *flags]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    result.check_returncode()
    outcome = json.loads(result.stdout.splitlines()[-1])
    print(json.dumps(outcome), flush=True)
    return outcome


def max_batch(args):
    best = None
    batch = 1
    while True:
        result = run_trial(args, batch)
        if result['status'] == 'oom':
            break
        best = result
        if batch == args.max_batch:
            return best, True
        batch = min(batch * 2, args.max_batch)

    low, high = (best['batch'] if best else 0), batch
    while high - low > 1:
        mid = (low + high) // 2
        result = run_trial(args, mid)
        if result['status'] == 'ok':
            low, best = mid, result
        else:
            high = mid
    return best, False


def main():
    parser = argparse.ArgumentParser(description='Probe batch size with the SwinIR-M ×4 training workload.')
    batch = parser.add_mutually_exclusive_group()
    batch.add_argument('--batch', type=int, help='Time one batch size')
    batch.add_argument('--max-batch', type=int, default=32, help='Upper bound for batch search (default: 32)')
    parser.add_argument('--steps', type=int, default=3, help='Timed steps per trial after one warm-up step')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--weights', type=Path, default=WEIGHT)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--activation-checkpoint', action='store_true')
    parser.add_argument('--fft', action='store_true', help='Probe the FFT variant')
    parser.add_argument('--trial', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(value is not None and value < 1 for value in (args.batch, args.max_batch, args.steps, args.trial)):
        parser.error('Batch sizes and steps must be positive.')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('The probe requires a CUDA device.')
    args.g_lr, args.d_lr = 1e-5, 1e-4

    if args.trial is not None:
        print(json.dumps(trial(args, device)))
        return 0
    if args.batch is not None:
        return int(run_trial(args, args.batch)['status'] != 'ok')
    best, reached_cap = max_batch(args)
    print(json.dumps({'largest_passing_batch': best['batch'] if best else None,
                      'reached_search_limit': reached_cap, 'largest_passing_result': best}))
    return int(best is None)


if __name__ == '__main__':
    raise SystemExit(main())
