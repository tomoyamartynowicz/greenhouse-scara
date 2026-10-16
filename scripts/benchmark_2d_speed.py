#!/usr/bin/env python3
"""Offline timing: thread selection, encoder/sampler breakdown and sampling budgets.

Uses the live Predictor and recorded camera/state inputs. Never connects to hardware.
Reduced sampling budgets are speed experiments, not validated deployment settings.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
VARIANTS = [f'{model}_{mode}' for model in ('act', 'dp', 'fm') for mode in ('rgb', 'rgbd')]


def load_histories(predictor, dataset, count):
    import h5py
    import numpy as np
    from greenhouse_scara_common.data import episode_paths
    paths = episode_paths(dataset)
    histories, sources = [], []
    for index in range(count):
        path = paths[index % len(paths)]
        with h5py.File(path) as data:
            frame = min(len(data['observations/qpos']) - 1, predictor.n_obs_steps - 1 + index * 7)
            history = []
            for t in range(frame - predictor.n_obs_steps + 1, frame + 1):
                t = max(t, 0)
                sample = dict(frames={}, depth_frames={}, depth_scales={})
                for key, spec in predictor.specs.items():
                    camera = spec.get('camera', key)
                    depth = spec['type'] == 'depth'
                    ds = data[f'observations/{"depth" if depth else "images"}/{camera}']
                    sample['depth_frames' if depth else 'frames'][camera] = ds[t]
                    if depth:
                        if bool(ds.attrs.get('aligned_to_rgb', False)) != bool(spec.get('aligned_to_rgb', False)):
                            raise ValueError('Recorded depth alignment differs from checkpoint')
                        sample['depth_scales'][camera] = float(ds.attrs['depth_scale'])
                qpos = np.concatenate([data['observations/qpos'][t],
                                       np.asarray(data['observations/scissor_rotation'][t]).reshape(1)])
                history.append((sample, qpos))
            histories.append(history)
            sources.append(dict(episode=path.name, frame=frame))
    return histories, sources


def worker(args):
    import numpy as np
    import torch
    from greenhouse_scara_common.inference import Predictor
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, mode = args.worker.split('_')
    path = args.checkpoints / args.worker / 'current' / ('policy_last.ckpt' if model == 'act' else 'latest.ckpt')
    predictor = Predictor(model, mode, path, args.device)
    histories, sources = load_histories(predictor, args.dataset, args.samples)
    policy = predictor.policy
    baseline_steps = getattr(policy, 'num_inference_steps', None)
    baseline_method = getattr(policy, 'integration_method', None)

    def sync():
        if predictor.device.type == 'cuda':
            torch.cuda.synchronize(predictor.device)

    def measure(threads, steps, method, samples):
        torch.set_num_threads(threads)
        if steps is not None:
            policy.num_inference_steps = steps
        if method is not None:
            policy.integration_method = method
        for i in range(args.warmup):
            torch.manual_seed(i)
            predictor.predict(histories[i % len(histories)])
        times = []
        for i in range(samples):
            torch.manual_seed(100 + i)
            sync()
            started = time.perf_counter()
            prediction = predictor.predict(histories[i % len(histories)])
            sync()
            times.append(1000 * (time.perf_counter() - started))
        row = dict(threads=threads, steps=steps, method=method, samples=samples,
                   network_evaluations=(steps * (2 if method == 'heun' else 1)) if steps else 1,
                   mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                   p95_ms=float(np.percentile(times, 95)), timings_ms=times,
                   predicted_actions=len(prediction))
        print(args.worker, json.dumps(row), flush=True)
        return row

    report = dict(variant=args.worker, checkpoint=str(path.resolve()), device=str(predictor.device),
                  gpu=torch.cuda.get_device_name(predictor.device) if predictor.device.type == 'cuda' else None,
                  torch_version=torch.__version__, warmup=args.warmup, observations=sources,
                  timing_scope='Live Predictor including preprocessing; excludes data loading and hardware I/O',
                  note='Reduced sampling budgets change predictions; no task-success validation.',
                  thread_probe=[], sampling_sweep=[])
    output = args.output / f'{args.worker}.json'
    args.output.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')

    # Probe with unchanged checkpoint settings, then confirm on more observations.
    for threads in args.threads:
        report['thread_probe'].append(measure(threads, baseline_steps, baseline_method, min(2, args.samples)))
        save()
    best = min(report['thread_probe'], key=lambda row: row['median_ms'])['threads']
    report['selected_threads'] = best
    report['baseline'] = measure(best, baseline_steps, baseline_method, args.samples)
    save()

    if model != 'act':
        # Separate diagnostic pass: syncs at boundaries slightly affect timings.
        # Normal timing above/below has no instrumentation inside the policy.
        phases = {}
        starts = {}

        def before(module, inputs):
            sync()
            starts['encoder'] = time.perf_counter()

        def after(module, inputs, outputs):
            sync()
            phases['encoder_ms'] = 1000 * (time.perf_counter() - starts['encoder'])

        original = policy.conditional_sample

        def sampling(*a, **kw):
            sync()
            started = time.perf_counter()
            result = original(*a, **kw)
            sync()
            phases['sampling_ms'] = 1000 * (time.perf_counter() - started)
            return result

        hooks = [policy.obs_encoder.register_forward_pre_hook(before),
                 policy.obs_encoder.register_forward_hook(after)]
        policy.conditional_sample = sampling
        try:
            sync()
            started = time.perf_counter()
            predictor.predict(histories[0])
            sync()
            phases['total_ms'] = 1000 * (time.perf_counter() - started)
        finally:
            policy.conditional_sample = original
            for hook in hooks:
                hook.remove()
        report['baseline_profile'] = phases
        save()

        cases = ([(n, 'heun') for n in (8, 4, 2, 1)] +
                 [(n, 'euler') for n in (8, 4, 2, 1)]) if model == 'fm' else [(n, None) for n in (50, 20, 10, 5, 1)]
        for steps, method in cases:
            report['sampling_sweep'].append(measure(best, steps, method, args.samples))
            save()
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints/2d')
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'datasets/green_house_dataset1_from11')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/benchmark_2d_speed')
    parser.add_argument('--worker', choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or min(args.threads) < 1:
        parser.error('samples/threads must be positive and warmup nonnegative')
    os.environ.setdefault('GREENHOUSE_MIN_EPISODE', '11')
    if args.worker:
        worker(args)
        return 0
    codes = {}
    for variant in args.variants:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                   '--checkpoints', str(args.checkpoints), '--dataset', str(args.dataset),
                   '--device', args.device, '--samples', str(args.samples), '--warmup', str(args.warmup),
                   '--output', str(args.output), '--threads', *map(str, args.threads)]
        codes[variant] = subprocess.run(command).returncode
    print('Results:', codes, flush=True)
    return int(any(codes.values()))


if __name__ == '__main__':
    raise SystemExit(main())
