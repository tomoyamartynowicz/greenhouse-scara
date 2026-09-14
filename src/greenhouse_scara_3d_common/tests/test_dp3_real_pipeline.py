from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from greenhouse_scara_3d_common.import_calibration import calibration_from_dump, import_dump
from greenhouse_scara_3d_common.pointcloud import PointCloudBuilder, sample_points
from greenhouse_scara_3d_common.dataset import PointCloudDataset
from test_3d import calibration, settings, config, episode


class RealPipelineTests(unittest.TestCase):
    def test_aligned_dump_uses_color_intrinsics_with_distortion(self):
        text = (SRC.parent / 'calibration/realsense_calibration-130322273198.txt').read_text()
        aligned = calibration_from_dump(text, 'camera_top', 640, 480, True)['cameras']['camera_top']
        native = calibration_from_dump(text, 'camera_top', 640, 480, False)['cameras']['camera_top']
        self.assertAlmostEqual(aligned['intrinsics']['fx'], 390.074615478516)
        self.assertAlmostEqual(native['intrinsics']['fx'], 383.847137451172)
        self.assertEqual(aligned['intrinsics']['distortion_model'], 'inverse_brown_conrady')
        self.assertNotEqual(aligned['intrinsics']['coeffs'], [0.] * 5)
        self.assertIsNone(aligned['to_reference'])
        with self.assertRaisesRegex(ValueError, 'profile'):
            calibration_from_dump(text, 'camera_top', 123, 456, True)

    def test_uncropped_keeps_far_and_max_code_values_and_every_pixel(self):
        opts = settings()
        opts.update(min_depth_m=0, max_depth_m=None, fps_candidates=None)
        builder = PointCloudBuilder(calibration(), opts)
        raw = np.full((8, 8), 4000, np.uint16)
        raw[0, 0] = 0
        raw[-1, -1] = 65535
        points = builder.dense({'camera_top': raw}, {'camera_top': .001})
        self.assertEqual(len(points), 63)
        self.assertAlmostEqual(float(points[:, 2].max()), 65.535, places=3)
        sampled = builder({'camera_top': raw}, {'camera_top': .001})
        np.testing.assert_array_equal(sampled, sample_points(points, 16, None))
        self.assertEqual(sampled.shape, (16, 3))

    def test_point_normalization_fits_only_training_clouds(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'episode_0.hdf5', Path(tmp) / 'episode_1.hdf5'
            episode(first); episode(second)
            with h5py.File(second, 'r+') as f:
                f['observations/depth/camera_top'][:] = 60000
            cfg = config('dp3', tmp)
            cfg['pointcloud'].update(normalization='limits', min_depth_m=0, max_depth_m=None)
            ds = PointCloudDataset([first], cfg)
            norm = ds.normalizer()
            points = ds[0]['obs']['point_cloud']
            normalized = norm['point_cloud'].normalize(points)
            self.assertTrue(np.isfinite(normalized.numpy()).all())
            np.testing.assert_allclose(normalized[:, :, :2].amin((0, 1)), [-1, -1], atol=1e-6)
            np.testing.assert_allclose(normalized[:, :, :2].amax((0, 1)), [1, 1], atol=1e-6)
            np.testing.assert_allclose(norm['point_cloud'].unnormalize(normalized), points, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
