#!/usr/bin/env python3
"""Batch-1 policy latency on preloaded dataset observations; no robot/cameras."""
import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / 'src', *(ROOT / 'src' / f'greenhouse_scara_{name}'
                             for name in ('act', 'diffusion_policy', 'flow_matching_policy'))):
    sys.path.insert(0, str(folder))

from greenhouse_scara_common.data import image_specs, read_images, read_padded


def load_policy(kind, checkpoint, device, steps):
    if kind == 'act':
        from policy import ACTPolicy
        with (checkpoint.parent / 'config.pkl').open('rb') as f:
            cfg = pickle.load(f)
        with (checkpoint.parent / 'dataset_stats.pkl').open('rb') as f:
            stats = pickle.load(f)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'policy' in state:
            state = state['policy']
        config = dict(cfg['policy_config'])
        config['pretrained_backbone'] = False
        config['num_queries'] = state['model.query_embed.weight'].shape[0]
        policy = ACTPolicy(config)
        policy.load_state_dict(state)
        specs = config['stream_specs']
        tensors = {k: torch.as_tensor(stats[k], dtype=torch.float32, device=device)
                   for k in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std')}

        def predict(obs):
            qpos = (obs['qpos'][:, 0] - tensors['qpos_mean']) / tensors['qpos_std']
            images = {k: obs[k][:, 0] for k in specs}
            return policy(qpos, images) * tensors['action_std'] + tensors['action_mean']

        metadata = {'history': 1, 'steps': None, 'sampler': 'ACT', 'unet_evaluations': None}
    else:
        import dill
        import hydra
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        # Load only policy weights, without rebuilding the optimizer or workspace.
        config = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=True))
        config.obs_encoder.pretrained = False
        policy = hydra.utils.instantiate(config)
        weights_key = 'ema_model' if cfg.training.use_ema else 'model'
        policy.load_state_dict(payload['state_dicts'][weights_key])
        if steps is not None:
            policy.num_inference_steps = steps
        specs = image_specs(cfg.shape_meta)
        sampler = getattr(policy, 'integration_method', 'DDIM')
        metadata = {'history': int(policy.n_obs_steps), 'steps': policy.num_inference_steps,
                    'sampler': sampler, 'weights': weights_key,
                    'unet_evaluations': policy.num_inference_steps * (2 if sampler == 'heun' else 1)}

        def predict(obs):
            return policy.predict_action(obs)['action']

    policy.to(device).eval()
    metadata['images'] = {k: list(v['shape']) for k, v in specs.items()}
    metadata['parameters'] = sum(p.numel() for p in policy.parameters())
    return policy, predict, specs, metadata


def observations(episode, specs, history, count, device):
    batches = []
    with h5py.File(episode, 'r') as root:
        length = len(root['observations/qpos'])
        if length < 1:
            raise ValueError('Episode has no samples')
        # Same current frame for each model; earlier history is padded if needed.
        frames = np.linspace(0, length - 1, min(count, length), dtype=int)
        for end in frames:
            start = int(end) - history + 1
            obs = {k: read_images(root, k, spec, start, history) for k, spec in specs.items()}
            obs['qpos'] = read_padded(root['observations/qpos'], start, history)
            batches.append({k: torch.as_tensor(v[None], dtype=torch.float32, device=device)
                            for k, v in obs.items()})
    return batches, frames.tolist()


@torch.inference_mode()
def benchmark(kind, checkpoint, episode, args, device):
    steps = getattr(args, f'{kind}_steps', None)
    policy, predict, specs, metadata = load_policy(kind, checkpoint, device, steps)
    inputs, frames = observations(episode, specs, metadata['history'], args.states, device)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    for i in range(args.warmup):
        prediction = predict(inputs[i % len(inputs)])
    synchronize()
    times = []
    for i in range(args.iterations):
        synchronize()
        start = time.perf_counter()
        prediction = predict(inputs[i % len(inputs)])
        synchronize()
        times.append((time.perf_counter() - start) * 1000)
    if not torch.isfinite(prediction).all():
        raise ValueError(f'{kind}: non-finite action prediction')
    result = dict(model=kind, checkpoint=str(checkpoint), device=str(device),
                  batch_size=1, frames=frames, iterations=args.iterations, warmup=args.warmup,
                  action_shape=list(prediction.shape), mean_ms=float(np.mean(times)),
                  median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                  calls_per_second=float(1000 / np.mean(times)), **metadata)
    print(f"\n{kind.upper()}: mean {result['mean_ms']:.2f} ms | "
          f"median {result['median_ms']:.2f} ms | p95 {result['p95_ms']:.2f} ms | "
          f"{result['calls_per_second']:.2f} calls/s", flush=True)
    print(f"  images={metadata['images']}, history={metadata['history']}, "
          f"action chunk={prediction.shape[1]}, sampler={metadata['sampler']}, "
          f"steps={metadata['steps']}, U-Net calls={metadata['unet_evaluations']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in ('act', 'dp', 'fm'):
        parser.add_argument(f'--{kind}', type=Path, help='Checkpoint file (optional)')
    parser.add_argument('--dataset', type=Path,
                        default=ROOT.parent / 'datasets/greenhouse_dummy_dataset')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:0')
    parser.add_argument('--states', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU threads')
    parser.add_argument('--dp-steps', type=int, help='Default: checkpoint setting')
    parser.add_argument('--fm-steps', type=int, help='Default: checkpoint setting')
    parser.add_argument('--output', type=Path, help='Optional JSON results file')
    args = parser.parse_args()
    if not any(getattr(args, k) for k in ('act', 'dp', 'fm')):
        parser.error('Provide at least one of --act, --dp, --fm')
    for key in ('states', 'warmup', 'iterations', 'threads', 'dp_steps', 'fm_steps'):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f'{key} must be positive')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('Only cpu and cuda devices are supported')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this Python environment')
    torch.set_num_threads(args.threads)
    episode = args.dataset.expanduser().resolve() / f'episode_{args.episode}.hdf5'
    hardware = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(f'Device: {device} ({hardware}), torch={torch.__version__}, CPU threads={args.threads}')
    print('Timing: preloaded batch-1 tensor -> action chunk; excludes disk I/O, resizing, '
          'CPU/GPU transfers, cameras and robot communication.', flush=True)
    results = []
    for kind in ('act', 'dp', 'fm'):
        path = getattr(args, kind)
        if path:
            torch.manual_seed(42)
            results.append(benchmark(kind, path.expanduser().resolve(), episode, args, device))
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'hardware': hardware, 'torch': torch.__version__,
                                         'cpu_threads': args.threads, 'episode': str(episode),
                                         'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
