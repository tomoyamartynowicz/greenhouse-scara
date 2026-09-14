#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from greenhouse_scara_3d_common.inspection import inspect_frame, export_inspection

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGB-D -> full cloud + DP3 input; no hardware required')
    parser.add_argument('--episode', type=Path, default=ROOT.parent / 'datasets/test_rgbd_single_joints/episode_0.hdf5')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'calibration/camera_top_d405_aligned.yaml')
    parser.add_argument('--config', type=Path, default=ROOT / 'src/greenhouse_scara_3d_diffusion_policy/config.yaml')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pointcloud_check')
    args = parser.parse_args()
    print('Reconstructing all valid points and sampling DP3 input...', flush=True)
    result = inspect_frame(args.episode, yaml.safe_load(args.calibration.read_text()),
                           yaml.safe_load(args.config.read_text())['pointcloud'], args.frame)
    print(export_inspection(result, args.output), flush=True)
    print(result['report'])
