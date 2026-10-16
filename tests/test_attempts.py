import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from greenhouse_scara_common.attempts import AttemptWriter, RecordedCamera
from greenhouse_scara_common.deployment import main


def sample():
    now = time.monotonic()
    return dict(frames={name: np.full((32, 48, 3), [240, 10, 30], dtype=np.uint8)
                        for name in ('camera_top', 'camera_bottom')},
                depth_frames={name: np.full((32, 48), 1000, dtype=np.uint16)
                              for name in ('camera_top', 'camera_bottom')},
                depth_scales={'camera_top': .001, 'camera_bottom': .001},
                depth_aligned={'camera_top': True, 'camera_bottom': True},
                image_times={'camera_top': now, 'camera_bottom': now}, image_time=now)


class AttemptTests(unittest.TestCase):
    def test_interrupt_finalizes_attempt_and_stops_both_devices(self):
        with tempfile.TemporaryDirectory() as temp:
            stopped = []
            class Robot:
                def __init__(self, **kwargs): pass
                def get_joint_vector(self): return np.array([.5, 0, 1, 0])
                def halt(self): stopped.append('robot')
                def close(self): pass
            class Scissor:
                def __init__(self, **kwargs): pass
                def read(self, **kwargs): return -1.
                def prepare_execution(self): pass
                def stop(self): stopped.append('scissor')
                def close(self): pass
            class Camera:
                def __init__(self, *args, **kwargs): pass
                def start(self): return self
                def read(self):
                    time.sleep(.01)
                    return sample()
                def close(self): pass
            predictor = SimpleNamespace(checkpoint=Path(temp) / 'fake.ckpt', device='cpu',
                                        specs={}, n_obs_steps=2, policy=SimpleNamespace())
            def interrupted(predictor, camera, *args):
                camera.capture_history(2)
                raise KeyboardInterrupt()
            with patch('greenhouse_scara_common.inference.Predictor', return_value=predictor), \
                 patch('greenhouse_scara_common.robot.TCSReadClient', Robot), \
                 patch('greenhouse_scara_common.robot.TCSMotionClient', Robot), \
                 patch('greenhouse_scara_common.scissor.ScissorClient', Scissor), \
                 patch('greenhouse_scara_common.live.MultiCameraSource', Camera), \
                 patch('greenhouse_scara_common.deployment.run_loop', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    main(['act', 'rgb', '--execute', '--record-attempt', 'test',
                          '--attempt-dir', temp, '--log-dir', temp])
            self.assertEqual(stopped, ['robot', 'scissor'])
            metadata = next(Path(temp).glob('*/attempt.json'))
            saved = json.loads(metadata.read_text())
            self.assertEqual(saved['action_count'], 15)
            self.assertEqual(saved['outcome'], 'interrupted')
            self.assertGreaterEqual(saved['recorded_frames'], 2)
            self.assertTrue((metadata.parent / 'camera_top.avi').is_file())

    def test_videos_state_and_optional_depth_are_readable(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = AttemptWriter(temp, 30, record_depth=True)
            for _ in range(3):
                writer.append(sample(), [.5, 0, 1, 0, -1], time.monotonic())
            writer.close()
            for name in ('camera_top', 'camera_bottom'):
                video = cv2.VideoCapture(str(Path(temp) / f'{name}.avi'))
                self.assertTrue(video.isOpened())
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                success, image = video.read()
                self.assertTrue(success)
                # Input RGB red becomes BGR red on decoding.
                self.assertGreater(image[..., 2].mean(), image[..., 0].mean())
                video.release()
            rows = [json.loads(line) for line in (Path(temp) / 'observations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['qpos'], [.5, 0, 1, 0, -1])
            with h5py.File(Path(temp) / 'depth.hdf5') as f:
                self.assertEqual(f['camera_top'].shape, (3, 32, 48))
                self.assertTrue(f['camera_top'].attrs['aligned_to_rgb'])
                self.assertAlmostEqual(f['camera_top'].attrs['depth_scale'], .001)

    def test_capture_continues_while_inference_is_busy(self):
        with tempfile.TemporaryDirectory() as temp:
            def read():
                time.sleep(.01)
                return sample()
            writer = AttemptWriter(temp, 30)
            stream = RecordedCamera(SimpleNamespace(read=read), lambda: np.zeros(5), writer, 2, .1).start()
            try:
                history = stream.capture_history(2)
                before = writer.frames
                time.sleep(.1)  # No reads by the inference consumer during this interval.
                self.assertGreater(writer.frames, before + 1)
                latest = stream.capture_history(2)
                self.assertGreater(latest[-1][0]['image_time'], history[-1][0]['image_time'])
                self.assertLessEqual(len(stream.history), 2)
            finally:
                stream.close()
            self.assertFalse((Path(temp) / 'depth.hdf5').exists())

    def test_camera_failure_propagates_to_consumer(self):
        with tempfile.TemporaryDirectory() as temp:
            def fail(): raise RuntimeError('camera disconnected')
            stream = RecordedCamera(SimpleNamespace(read=fail), lambda: np.zeros(5),
                                    AttemptWriter(temp, 30), 2, .1).start()
            with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
                stream.capture_history(2)
            with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
                stream.close()
            self.assertTrue(stream.writer.telemetry.closed)


if __name__ == '__main__':
    unittest.main()
