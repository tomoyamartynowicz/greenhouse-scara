#!/usr/bin/env python3
"""Prepare the same single-camera XYZRGB sidecars for ACT3D, DP3 and Flow3D."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from greenhouse_scara_3d_common.runtime import load_config
    from greenhouse_scara_3d_common.cache import prepare_episode
    from greenhouse_scara_common.data import episode_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-episodes', type=int)
    args, remaining = parser.parse_known_args()
    if args.workers < 1 or (args.max_episodes is not None and args.max_episodes < 1):
        parser.error('workers/max-episodes must be positive')
    cfg = load_config(ROOT / 'src/greenhouse_scara_3d_diffusion_policy/config_greenhouse.yaml', remaining)
    os.environ.setdefault('GREENHOUSE_MIN_EPISODE', '11')
    paths = episode_paths(cfg['dataset_dir'])[:args.max_episodes]
    print(f"Preparing {len(paths)} episodes -> {cfg.get('pointcloud_cache_dir')}", flush=True)
    if args.workers == 1:
        results = map(partial(prepare_episode, cfg=cfg), paths)
        for index, result in enumerate(results, 1):
            print(index, len(paths), *result, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, result in enumerate(pool.map(partial(prepare_episode, cfg=cfg), paths), 1):
                print(index, len(paths), *result, flush=True)


if __name__ == '__main__':
    main()
