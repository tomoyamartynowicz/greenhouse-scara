import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from greenhouse_scara_common.deployment import main


class Deployment3DTests(unittest.TestCase):
    def test_3d_uses_checkpoint_camera_and_action_defaults_without_motion(self):
        for model, count in [('act3d', 15), ('dp3', 8), ('fm3', 8)]:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                camera = Mock()
                predictor = SimpleNamespace(checkpoint=Path(tmp) / 'latest.ckpt', device='cpu',
                    policy=SimpleNamespace(), cfg={'pointcloud': {'cameras': ['camera_top']},
                    'calibration': {'cameras': {'camera_top': {'serial': 'saved-camera'}}}},
                    camera_source=Mock(return_value=camera))
                robot = Mock(); robot.get_joint_vector.return_value = np.array([.5, 0, 1, 0])
                scissor = Mock(); scissor.read.return_value = -1.
                def loop(pred, cam, rob, sci, settings, args, log):
                    self.assertEqual(args.action_count, count)
                    self.assertFalse(args.execute)
                    self.assertIs(cam, camera)
                    return 'chunk_limit'
                with patch('greenhouse_scara_3d_common.inference.Predictor3D', return_value=predictor), \
                     patch('greenhouse_scara_common.live.MultiCameraSource') as ordinary_camera, \
                     patch('greenhouse_scara_common.robot.TCSReadClient', return_value=robot), \
                     patch('greenhouse_scara_common.robot.TCSMotionClient') as motion, \
                     patch('greenhouse_scara_common.scissor.ScissorClient', return_value=scissor), \
                     patch('greenhouse_scara_common.deployment.run_loop', side_effect=loop):
                    main([model, 'pointcloud', '--log-dir', tmp])
                    ordinary_camera.assert_not_called()
                    motion.assert_not_called()
                    scissor.set_rotation.assert_not_called()
                predictor.camera_source.assert_called_once_with(fps=30)
                camera.start.assert_called_once()
                camera.close.assert_called_once()
                record = json.loads(next(Path(tmp).glob('*.jsonl')).read_text().splitlines()[0])
                self.assertEqual(record['calibration'], predictor.cfg['calibration'])


if __name__ == '__main__':
    unittest.main()
