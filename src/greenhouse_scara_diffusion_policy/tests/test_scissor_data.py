"""Check roll command timing, sequence padding, validation and normalization."""
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

DP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DP), str(DP.parent)]
from greenhouse_diffusion_policy.dataset.scara_image_dataset import ScaraImageDataset
from greenhouse_diffusion_policy.env_runner.scara_image_runner import ScaraImageRunner


class ScissorDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'episode_11.hdf5'
        self.qpos = np.arange(24, dtype=np.float32).reshape(6, 4)
        self.roll = np.arange(6, dtype=np.float32) * .1
        self.target = np.arange(6, dtype=np.float32) * .2 + 1
        self.meta = {'obs': {'camera_top_rgb': {'camera': 'camera_top', 'type': 'rgb',
                                              'shape': [3, 32, 32]},
                             'qpos': {'type': 'low_dim', 'shape': [5]}},
                     'action': {'shape': [5]}}
        with h5py.File(self.path, 'w') as root:
            root['observations/qpos'] = self.qpos
            root['observations/scissor_rotation'] = self.roll
            root['actions/scissor_rotation'] = self.target[:, None]
            root['observations/images/camera_top'] = np.zeros((6, 32, 32, 3), np.uint8)

    def dataset(self, **kwargs):
        return ScaraImageDataset(self.meta, self.temp.name, horizon=4, n_obs_steps=2,
                                 pad_before=1, pad_after=2, include_scissor_roll=True, **kwargs)

    def test_command_is_unshifted_and_boundaries_repeat(self):
        dataset = self.dataset(action_offset=2)
        dataset.samples = [(0, -1), (0, 1), (0, 4)]
        for index, start in enumerate((-1, 1, 4)):
            sample = dataset[index]
            obs_rows = np.clip(np.arange(start, start + 2), 0, 5)
            action_rows = np.clip(np.arange(start, start + 4), 0, 5)
            np.testing.assert_array_equal(sample['obs']['qpos'],
                                          np.column_stack([self.qpos, self.roll])[obs_rows])
            np.testing.assert_array_equal(sample['action'][:, :4],
                                          self.qpos[np.minimum(action_rows + 2, 5)])
            np.testing.assert_array_equal(sample['action'][:, 4], self.target[action_rows])
            self.assertEqual(tuple(sample['action'].shape), (4, 5))

    def test_stored_robot_commands_are_preserved(self):
        with h5py.File(self.path, 'a') as root:
            root['action'] = self.qpos + 100
        dataset = self.dataset()
        dataset.samples = [(0, 1)]
        np.testing.assert_array_equal(dataset[0]['action'],
                                      np.column_stack([self.qpos + 100, self.target])[1:5])

    def test_invalid_roll_or_dimensions_fail(self):
        for key in ('observations/scissor_rotation', 'actions/scissor_rotation'):
            for values in (None, np.zeros(5), np.full(6, np.nan)):
                with self.subTest(key=key, values=values):
                    with h5py.File(self.path, 'a') as root:
                        del root[key]
                        if values is not None:
                            root[key] = values
                    with self.assertRaises(ValueError):
                        self.dataset()
                    with h5py.File(self.path, 'a') as root:
                        if key in root:
                            del root[key]
                        root[key] = self.roll if key.startswith('observations') else self.target
        with self.assertRaisesRegex(ValueError, 'include_scissor_roll'):
            ScaraImageDataset(self.meta, self.temp.name)

    def test_train_only_normalizer_and_offline_runner_support_five_joints(self):
        # Different episode lengths must work, with a held-out extreme roll range.
        other = self.path.with_name('episode_12.hdf5')
        with h5py.File(other, 'w') as root:
            root['observations/qpos'] = self.qpos[:4]
            root['observations/scissor_rotation'] = np.full(4, 1000, np.float32)
            root['actions/scissor_rotation'] = np.full(4, 2000, np.float32)
            root['observations/images/camera_top'] = np.zeros((4, 32, 32, 3), np.uint8)
        dataset = self.dataset(val_ratio=.5, seed=0)
        self.assertEqual(dataset.train_mask.tolist(), [True, False])
        normalizer = dataset.get_normalizer()
        stats = normalizer['action'].get_input_stats()
        self.assertAlmostEqual(float(stats['max'][4]), float(self.target.max()))
        sample = dataset[0]['action']
        np.testing.assert_allclose(normalizer['action'].unnormalize(
            normalizer['action'].normalize(sample)), sample, atol=1e-5)
        runner = ScaraImageRunner(self.temp.name, self.meta, self.temp.name,
                                 horizon=4, pad_before=1, pad_after=2, n_obs_steps=2,
                                 n_action_steps=2, val_ratio=.5, seed=0,
                                 include_scissor_roll=True)
        self.assertEqual(tuple(runner.dataset[0]['action'].shape), (4, 5))
        np.testing.assert_array_equal(runner.dataset[0]['action'][:, 4], 2000)


if __name__ == '__main__':
    unittest.main()
