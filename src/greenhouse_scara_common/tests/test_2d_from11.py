"""Check the six 2D jobs' shared episode selection and five-dimensional data."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SRC), str(SRC / 'greenhouse_scara_diffusion_policy'),
                str(SRC / 'greenhouse_scara_flow_matching_policy')]
from greenhouse_scara_common.data import episode_paths
from greenhouse_diffusion_policy.dataset.scara_image_dataset import ScaraImageDataset as DPData
from greenhouse_flow_matching_policy.dataset.scara_image_dataset import ScaraImageDataset as FMData


class From11Tests(unittest.TestCase):
    def test_selection_rgbd_views_and_unshifted_scissor_commands(self):
        meta = {'obs': {}, 'action': {'shape': [5]}}
        for camera in ('camera_top', 'camera_bottom'):
            meta['obs'][camera + '_rgb'] = {'camera': camera, 'type': 'rgb', 'shape': [3, 8, 8]}
            meta['obs'][camera + '_depth'] = {
                'camera': camera, 'type': 'depth', 'shape': [2, 8, 8],
                'depth_normalization_m': 1.0, 'aligned_to_rgb': True}
        meta['obs']['qpos'] = {'type': 'low_dim', 'shape': [5]}
        with tempfile.TemporaryDirectory() as directory:
            for number in (10, 11, 12):
                with h5py.File(Path(directory) / f'episode_{number}.hdf5', 'w') as root:
                    root['observations/qpos'] = np.arange(24, dtype=np.float32).reshape(6, 4)
                    root['observations/scissor_rotation'] = np.arange(6, dtype=np.float32)
                    root['actions/scissor_rotation'] = np.arange(6, dtype=np.float32) + number
                    for camera in ('camera_top', 'camera_bottom'):
                        root[f'observations/images/{camera}'] = np.zeros((6, 8, 8, 3), np.uint8)
                        depth = root.create_dataset(f'observations/depth/{camera}',
                                                    data=np.full((6, 8, 8), 1000, np.uint16))
                        depth.attrs['depth_scale'] = .0001
                        depth.attrs['aligned_to_rgb'] = True
            with patch.dict(os.environ, {'GREENHOUSE_MIN_EPISODE': '11'}):
                self.assertEqual([path.name for path in episode_paths(directory)],
                                 ['episode_11.hdf5', 'episode_12.hdf5'])
                for cls in (DPData, FMData):
                    data = cls(meta, directory, horizon=4, n_obs_steps=2,
                               include_scissor_roll=True, val_ratio=.5)
                    self.assertEqual([path.name for path in data.episode_paths],
                                     ['episode_11.hdf5', 'episode_12.hdf5'])
                    data.samples = [(0, 0)]
                    sample = data[0]
                    self.assertEqual(set(sample['obs']), set(meta['obs']))
                    self.assertEqual(tuple(sample['action'].shape), (4, 5))
                    np.testing.assert_array_equal(sample['action'][:, 4],
                                                  np.arange(4, dtype=np.float32) + 11)


if __name__ == '__main__':
    unittest.main()
