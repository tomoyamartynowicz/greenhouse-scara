#!/usr/bin/env python3
"""Run ACT on recorded observations, without connecting to robot hardware."""
import argparse
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenhouse_scara_common.data import episode_paths
from policy import ACTPolicy
from utils import EpisodicDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='policy_best.ckpt')
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--expected-modality', choices=('rgb', 'rgbd'))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    with (ckpt_dir / 'config.pkl').open('rb') as source:
        config = pickle.load(source)
    if args.expected_modality:
        meta = config['inputs_config']['shape_meta']
        actual = {key for key in meta['obs'] if key != 'qpos'}
        expected = {'camera_top_rgb', 'camera_bottom_rgb'}
        if args.expected_modality == 'rgbd':
            expected |= {'camera_top_depth', 'camera_bottom_depth'}
        if actual != expected or meta['obs']['qpos']['shape'] != [5] or meta['action']['shape'] != [5]:
            parser.error('Checkpoint inputs/actions do not match the requested 2D variant')
    with (ckpt_dir / 'dataset_stats.pkl').open('rb') as source:
        stats = pickle.load(source)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ckpt_dir / checkpoint
    state = torch.load(checkpoint, map_location='cpu')
    if 'policy' in state:
        state = state['policy']
    config['policy_config']['num_queries'] = state['model.query_embed.weight'].shape[0]
    config['policy_config']['pretrained_backbone'] = False
    builder = ['eval_offline', '--ckpt_dir', str(ckpt_dir), '--policy_class', 'ACT',
               '--task_name', config['task_name'], '--seed', str(config['seed']),
               '--num_epochs', str(config['num_epochs'])]
    with patch.object(sys, 'argv', builder):
        policy = ACTPolicy(config['policy_config'])
    device = torch.device(args.device)
    policy.load_state_dict(state)
    policy.to(device).eval()
    paths = episode_paths(args.dataset_dir)
    dataset = EpisodicDataset(paths, config['inputs_config']['shape_meta'], stats,
                              config['policy_config']['num_queries'],
                              config['inputs_config'].get('action_source', 'auto'),
                              config['inputs_config'].get('action_offset', 1),
                              config['inputs_config'].get('include_scissor_roll', False))
    squared_error = np.zeros(len(stats['action_mean']), dtype=np.float64)
    total = 0
    with torch.inference_mode():
        for index in range(args.samples):
            images, qpos, target, padded = dataset[index % len(dataset)]
            images = {key: value.unsqueeze(0).to(device) for key, value in images.items()}
            predicted = policy(qpos.unsqueeze(0).to(device), images)[0].cpu().numpy()
            truth = target.numpy()
            valid = ~padded.numpy()
            error = (predicted[valid] - truth[valid]) * stats['action_std']
            squared_error += np.square(error).sum(axis=0)
            total += int(valid.sum())
    print(json.dumps({'checkpoint': str(checkpoint), 'dataset': str(args.dataset_dir),
                      'samples': args.samples, 'valid_actions': total,
                      'mse_per_action': (squared_error / total).tolist(),
                      'mse': float(squared_error.sum() / (total * len(squared_error)))}, indent=2))


if __name__ == '__main__':
    main()
