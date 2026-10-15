"""Check measured roll vs commanded roll, timing, padding and train-only stats."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ACT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ACT), str(ACT.parent)]
from utils import EpisodicDataset, load_data, read_joint_arrays


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
            root['actions/scissor_rotation'] = self.target
            root['observations/images/camera_top'] = np.zeros((6, 32, 32, 3), np.uint8)

    def test_roll_observation_and_command_are_distinct_and_command_is_unshifted(self):
        with h5py.File(self.path, 'r') as root:
            state, actions = read_joint_arrays(root, include_scissor_roll=True)
            np.testing.assert_array_equal(state, np.column_stack([self.qpos, self.roll]))
            np.testing.assert_array_equal(actions[:, :4], self.qpos[[1, 2, 3, 4, 5, 5]])
            np.testing.assert_array_equal(actions[:, 4], self.target)
            old_state, old_actions = read_joint_arrays(root)
            self.assertEqual(old_state.shape, (6, 4))
            self.assertEqual(old_actions.shape, (6, 4))

    def test_stored_robot_actions_and_five_dimensional_chunk_padding(self):
        with h5py.File(self.path, 'a') as root:
            root['action'] = self.qpos + 100
        stats = {key: np.zeros(5, np.float32) if key.endswith('mean') else np.ones(5, np.float32)
                 for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std')}
        dataset = EpisodicDataset([self.path], self.meta, stats, 4, include_scissor_roll=True)
        with patch.object(np.random, 'randint', return_value=4):
            images, state, actions, pad = dataset[0]
        np.testing.assert_array_equal(state, np.r_[self.qpos[4], self.roll[4]])
        np.testing.assert_array_equal(actions[:2, :4], self.qpos[4:] + 100)
        np.testing.assert_array_equal(actions[:2, 4], self.target[4:])
        np.testing.assert_array_equal(pad, [False, False, True, True])
        self.assertEqual(tuple(actions.shape), (4, 5))
        self.assertEqual(set(images), {'camera_top_rgb'})

    def test_missing_misaligned_and_nonfinite_roll_fail(self):
        for value in (None, np.zeros(5), np.full(6, np.nan)):
            with self.subTest(value=value):
                with h5py.File(self.path, 'a') as root:
                    if 'actions/scissor_rotation' in root:
                        del root['actions/scissor_rotation']
                    if value is not None:
                        root['actions/scissor_rotation'] = value
                with h5py.File(self.path, 'r') as root, self.assertRaises(ValueError):
                    read_joint_arrays(root, include_scissor_roll=True)

    def test_fifth_joint_normalization_uses_training_episodes_only(self):
        other = self.path.with_name('episode_12.hdf5')
        other.write_bytes(self.path.read_bytes())
        with h5py.File(other, 'a') as root:
            root['observations/scissor_rotation'][:] = 1000
            root['actions/scissor_rotation'][:] = 2000
        cfg = {'shape_meta': self.meta, 'include_scissor_roll': True}
        with patch.object(np.random, 'permutation', return_value=np.array([0, 1])):
            train, validation, stats = load_data(self.temp.name, 2, ['camera_top_rgb'],
                                                1, 1, 4, cfg, num_workers=0)
        self.assertEqual(train.dataset.paths, [self.path])
        self.assertEqual(validation.dataset.paths, [other])
        self.assertAlmostEqual(float(stats['qpos_mean'][4]), float(self.roll.mean()))
        self.assertAlmostEqual(float(stats['action_mean'][4]), float(self.target.mean()))
        self.assertEqual(stats['action_std'].shape, (5,))


if __name__ == '__main__':
    unittest.main()
