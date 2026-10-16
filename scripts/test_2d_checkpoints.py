#!/usr/bin/env python3
"""Smoke-test the actual live inference path on recorded observations (no hardware)."""
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
VARIANTS = tuple(f'{model}_{mode}' for model in ('act', 'dp', 'fm') for mode in ('rgb', 'rgbd'))


def evaluate(args, variant):
    import h5py
    import numpy as np
    import torch
    from greenhouse_scara_common.inference import Predictor
    from greenhouse_scara_common.data import episode_paths, action_dataset
    model, modality = variant.split('_')
    checkpoint = args.checkpoints / variant / 'current' / ('policy_last.ckpt' if model == 'act' else 'latest.ckpt')
    predictor = Predictor(model, modality, checkpoint, args.device, args.num_inference_steps)
    paths = episode_paths(args.dataset)
    errors, latencies = [], []
    predictions = []
    task = predictor.cfg['inputs_config'] if model == 'act' else predictor.cfg.task
    for index in range(args.samples):
        path = paths[index % len(paths)]
        with h5py.File(path) as data:
            count = len(data['observations/qpos'])
            frame = min(count - 1, predictor.n_obs_steps - 1 + index * 7)
            history = []
            for t in range(frame - predictor.n_obs_steps + 1, frame + 1):
                t = max(0, t)
                sample = dict(frames={}, depth_frames={}, depth_scales={})
                for key, spec in predictor.specs.items():
                    camera = spec.get('camera', key)
                    group = 'images' if spec['type'] == 'rgb' else 'depth'
                    ds = data[f'observations/{group}/{camera}']
                    sample['frames' if group == 'images' else 'depth_frames'][camera] = ds[t]
                    if group == 'depth':
                        if bool(ds.attrs.get('aligned_to_rgb', False)) != bool(spec.get('aligned_to_rgb', False)):
                            raise ValueError('Recorded depth alignment differs from checkpoint')
                        sample['depth_scales'][camera] = float(ds.attrs['depth_scale'])
                qpos = np.concatenate([data['observations/qpos'][t], np.asarray(data['observations/scissor_rotation'][t]).reshape(1)])
                history.append((sample, qpos))
            if index == 0:
                for _ in range(args.warmup):
                    predictor.predict(history)
            if predictor.device.type == 'cuda':
                torch.cuda.synchronize(predictor.device)
            started = time.monotonic()
            prediction = predictor.predict(history)
            if predictor.device.type == 'cuda':
                torch.cuda.synchronize(predictor.device)
            latencies.append(1000 * (time.monotonic() - started))
            targets = action_dataset(data, task.get('action_source', 'auto'), int(task.get('action_offset', 1)))
            length = min(len(prediction), count - frame)
            truth = np.concatenate([targets[frame:frame + length],
                                    data['actions/scissor_rotation'][frame:frame + length].reshape(-1, 1)], axis=1)
            errors.extend((prediction[:length] - truth) ** 2)
            predictions.append(dict(episode=path.name, frame=frame, first_action=prediction[0].tolist()))
    result = dict(variant=variant, checkpoint=str(checkpoint.resolve()), samples=args.samples,
                  warmup=args.warmup, device=str(predictor.device), torch_version=torch.__version__,
                  cpu_threads=torch.get_num_threads(), dataset=str(args.dataset.resolve()),
                  timing_scope='predict(history): image preprocessing, model inference and CPU action output; excludes checkpoint loading, dataset I/O and robot/camera communication',
                  predicted_actions=len(prediction),
                  num_inference_steps=getattr(predictor.policy, 'num_inference_steps', None),
                  image_shapes={key: list(spec['shape']) for key, spec in predictor.specs.items()},
                  latency_ms=dict(mean=float(np.mean(latencies)), median=float(np.median(latencies)),
                                  p95=float(np.percentile(latencies, 95)), min=float(np.min(latencies)),
                                  max=float(np.max(latencies))),
                  action_names=['J1', 'J2', 'J3', 'J4', 'scissor_roll'],
                  mse_per_action=np.mean(errors, axis=0).tolist(), inference_ms=latencies,
                  predictions=predictions, note='Smoke test on recorded data; not a held-out performance estimate.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'{variant}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints/2d')
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'datasets/green_house_dataset1_from11')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=0, help='Untimed predictions before measuring the first sample')
    parser.add_argument('--num-inference-steps', type=int)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/checkpoint_tests')
    parser.add_argument('--worker', choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative')
    os.environ.setdefault('GREENHOUSE_MIN_EPISODE', '11')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if args.worker:
        evaluate(args, args.worker)
        return 0
    results = {}
    for variant in args.variants:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                   '--checkpoints', str(args.checkpoints), '--dataset', str(args.dataset),
                   '--device', args.device, '--samples', str(args.samples), '--warmup', str(args.warmup),
                   '--output', str(args.output)]
        if args.num_inference_steps is not None:
            command += ['--num-inference-steps', str(args.num_inference_steps)]
        # Separate processes release model RAM between variants.
        results[variant] = subprocess.run(command).returncode
    print('Results:', results)
    return int(any(results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
