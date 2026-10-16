import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from test_3d import config, episode
from greenhouse_scara_3d_common.cache import prepare_episode, validate_cache
from greenhouse_scara_3d_common.dataset import PointCloudDataset
from greenhouse_scara_3d_common.inference import Predictor3D
from greenhouse_scara_3d_common.runtime import train


def scissor_episode(path):
    episode(path)
    with h5py.File(path, 'r+') as f:
        f.create_dataset('observations/scissor_rotation', data=np.linspace(-1, 1, 8, dtype=np.float32))
        f.create_dataset('actions/scissor_rotation', data=np.linspace(1, -1, 8, dtype=np.float32))


class Greenhouse3DTests(unittest.TestCase):
    def test_camera_mismatch_requires_explicit_approximation_and_same_resolution(self):
        from greenhouse_scara_3d_common.live import CalibratedCameraSource
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config('dp3', tmp)
            actual = copy.deepcopy(cfg['calibration']['cameras']['camera_top'])
            actual['intrinsics']['fx'] += 1
            frame = SimpleNamespace(get_depth_frame=lambda: object())
            source = SimpleNamespace(pipeline=SimpleNamespace(wait_for_frames=lambda timeout: frame),
                                     align=None, serial='test-only', align_depth=True, color_name='camera_top')
            with patch('greenhouse_scara_3d_common.live.MultiCameraSource') as cls, \
                 patch('greenhouse_scara_3d_common.export_calibration.calibration_from_frame', return_value=actual):
                cls.return_value.sources = [source]
                with self.assertRaisesRegex(ValueError, 'intrinsics differ'):
                    CalibratedCameraSource(cfg).start()
                cfg['calibration']['provenance'] = dict(approximate_intrinsics=True, user_accepted_approximation=True)
                with self.assertWarnsRegex(UserWarning, 'approximate intrinsics'):
                    camera = CalibratedCameraSource(cfg).start()
                camera.close()
                actual['width'] += 1
                with self.assertRaisesRegex(ValueError, 'profile differs'):
                    CalibratedCameraSource(cfg).start()

    def test_cache_matches_online_points_and_rejects_stale_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episode_0.hdf5'; scissor_episode(path)
            cfg = config('dp3', tmp)
            cfg.update(include_scissor_roll=True)
            cfg['pointcloud'].update(use_color=True, normalization='limits')
            online = PointCloudDataset([path], cfg)[3]
            cfg['pointcloud_cache_dir'] = str(Path(tmp) / 'cache')
            prepare_episode(path, cfg)
            self.assertEqual(prepare_episode(path, cfg)[1], 'existing')
            with patch('greenhouse_scara_3d_common.pointcloud.PointCloudBuilder.from_hdf5', side_effect=AssertionError('Unexpected FPS')):
                dataset = PointCloudDataset([path], cfg)
                cached = dataset[3]
                norm = dataset.normalizer()
            np.testing.assert_array_equal(cached['obs']['point_cloud'], online['obs']['point_cloud'])
            self.assertEqual(cached['obs']['agent_pos'].shape, (2, 5))
            with h5py.File(path) as f:
                np.testing.assert_array_equal(cached['action'][:6, 4], f['actions/scissor_rotation'][2:])
                np.testing.assert_array_equal(cached['obs']['agent_pos'][:, 4], f['observations/scissor_rotation'][2:4])
            self.assertEqual(norm['action'].get_input_stats()['min'].shape, (5,))
            changed = copy.deepcopy(cfg); changed['pointcloud']['max_depth_m'] = 2.
            with self.assertRaisesRegex(ValueError, 'cache differs'):
                validate_cache(path, changed)
            with h5py.File(path, 'r+') as f:
                f['observations/depth/camera_top'][0, 0, 0] = 0
            with self.assertRaisesRegex(ValueError, 'cache differs'):
                validate_cache(path, cfg)

    def test_missing_scissor_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episode_0.hdf5'; episode(path)
            cfg = config('dp3', tmp); cfg['include_scissor_roll'] = True
            with self.assertRaisesRegex(ValueError, 'scissor_rotation'):
                PointCloudDataset([path], cfg)

    def test_three_models_train_reload_and_predict_five_actions(self):
        for model, alias in [('act3d', 'act3d'), ('dp3', 'dp3'), ('flow3d', 'fm3')]:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                paths = [Path(tmp) / f'episode_{i}.hdf5' for i in range(2)]
                for path in paths:
                    scissor_episode(path)
                cfg = config(model, tmp)
                cfg.update(include_scissor_roll=True, checkpoint_every=1, keep_checkpoints=2,
                           pointcloud_cache_dir=str(Path(tmp) / 'cache'))
                cfg['pointcloud'].update(use_color=True, normalization='limits')
                for path in paths:
                    prepare_episode(path, cfg)
                checkpoint = train(cfg)
                predictor = Predictor3D(alias, checkpoint, 'cpu')
                dataset = PointCloudDataset(paths, cfg)
                sample = dataset[3]
                history = []
                with h5py.File(paths[0]) as f:
                    for t in range(4 - cfg['n_obs_steps'], 4):
                        packet = dict(depth_frames={'camera_top': f['observations/depth/camera_top'][t]},
                                      depth_scales={'camera_top': .001}, depth_aligned={'camera_top': True},
                                      frames={'camera_top': f['observations/images/camera_top'][t]})
                        history.append((packet, np.r_[f['observations/qpos'][t], f['observations/scissor_rotation'][t]]))
                # Same RNG -> cache-based model input and raw-camera adapter must agree.
                torch.manual_seed(10)
                predicted = predictor.predict(history)
                torch.manual_seed(10)
                with torch.inference_mode():
                    direct = predictor.policy.predict_action({k: v[None] for k, v in sample['obs'].items()})
                expected = direct['action_pred' if model == 'act3d' else 'action'][0].numpy()
                np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-5)
                self.assertEqual(predicted.shape[-1], 5)
                self.assertTrue(np.isfinite(predicted).all())
                self.assertTrue((checkpoint.parent / 'epoch_000000.ckpt').is_file())


if __name__ == '__main__':
    unittest.main()
