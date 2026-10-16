#!/usr/bin/env python3
"""Run trained 3D checkpoints on local demonstrations without connecting to hardware."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def evaluate(args, model):
    import numpy as np
    import torch
    from greenhouse_scara_3d_common.inference import Predictor3D
    from benchmark_2d_speed import load_histories
    checkpoint = args.checkpoint or args.checkpoints / model / 'latest.ckpt'
    predictor = Predictor3D(model, checkpoint, args.device, args.num_inference_steps)
    histories, sources = load_histories(predictor, args.dataset, args.samples)
    for _ in range(args.warmup):
        predictor.predict(histories[0])
    times, predictions = [], []
    for history in histories:
        if predictor.device.type == 'cuda':
            torch.cuda.synchronize(predictor.device)
        started = time.perf_counter()
        actions = predictor.predict(history)
        if predictor.device.type == 'cuda':
            torch.cuda.synchronize(predictor.device)
        times.append(1000 * (time.perf_counter() - started))
        predictions.append(actions[0].tolist())
    result = dict(model=model, checkpoint=str(predictor.checkpoint), device=str(predictor.device),
                  action_names=['J1', 'J2', 'J3', 'J4', 'scissor_roll'], prediction_shape=list(actions.shape),
                  inference_ms=times, mean_ms=float(np.mean(times)), observations=sources,
                  first_actions=predictions, pointcloud=predictor.cfg['pointcloud'],
                  calibration=predictor.cfg['calibration'],
                  note='Technical check and timing including pointcloud creation/FPS. No hardware or task-success evaluation.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'{model}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=('act3d', 'dp3', 'fm3'), default=['act3d', 'dp3', 'fm3'])
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--checkpoint', type=Path, help='Explicit checkpoint for one selected variant')
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'datasets/green_house_dataset1_from11')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-inference-steps', type=int)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/checkpoint_tests_3d')
    parser.add_argument('--worker', choices=('act3d', 'dp3', 'fm3'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error('samples must be positive and warmup nonnegative')
    if args.checkpoint and not args.worker and len(args.variants) != 1:
        parser.error('--checkpoint requires exactly one --variants entry')
    os.environ.setdefault('GREENHOUSE_MIN_EPISODE', '11')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if args.worker:
        evaluate(args, args.worker)
        return 0
    codes = {}
    for model in args.variants:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', model,
                   '--checkpoints', str(args.checkpoints), '--dataset', str(args.dataset),
                   '--device', args.device, '--samples', str(args.samples), '--warmup', str(args.warmup),
                   '--output', str(args.output)]
        if args.checkpoint:
            command += ['--checkpoint', str(args.checkpoint)]
        if args.num_inference_steps is not None:
            command += ['--num-inference-steps', str(args.num_inference_steps)]
        codes[model] = subprocess.run(command).returncode
    print('Results:', codes)
    return int(any(codes.values()))


if __name__ == '__main__':
    raise SystemExit(main())
