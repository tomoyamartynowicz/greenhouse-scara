"""Raw RealSense observations to the same calibrated point-cloud input as the loader."""
from collections import deque
import numpy as np
import torch
from greenhouse_scara_common.live import MultiCameraSource
from .pointcloud import PointCloudBuilder


class LivePointCloudSource:
    """Camera-only adapter. The caller supplies current joints; it never commands the robot."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.builder = PointCloudBuilder(cfg['calibration'], cfg['pointcloud'])
        names = cfg['pointcloud']['cameras']
        cams = cfg['calibration']['cameras']
        sizes = {(cams[n]['width'], cams[n]['height']) for n in names}
        if len(sizes) != 1: raise ValueError('Live cameras currently require the same capture resolution')
        specs = {n+'_depth': dict(camera=n, type='depth', aligned_to_rgb=cams[n]['aligned_to_rgb']) for n in names}
        self.source = MultiCameraSource(specs, serials={n:cams[n]['serial'] for n in names}, size=next(iter(sizes)))
        self.history = deque(maxlen=cfg['n_obs_steps'])

    def start(self):
        self.source.start()
        try:
            # Verify the processed frame profile instead of assuming calibration still matches.
            from .export_calibration import calibration_from_frame
            for source in self.source.sources:
                frames = source.pipeline.wait_for_frames(10000)
                if source.align: frames = source.align.process(frames)
                actual = calibration_from_frame(frames.get_depth_frame(), source.serial, source.align_depth)
                saved = self.cfg['calibration']['cameras'][source.color_name]
                for key in ('fx', 'fy', 'cx', 'cy'):
                    if not np.isclose(actual['intrinsics'][key], saved['intrinsics'][key], rtol=1e-5):
                        raise ValueError(f'{source.color_name}: live intrinsics differ from training calibration')
                if (actual['width'], actual['height'], actual['intrinsics']['distortion_model']) != (saved['width'], saved['height'], saved['intrinsics']['distortion_model']):
                    raise ValueError('Live image profile differs from calibration')
                if not np.allclose(actual['intrinsics']['coeffs'], saved['intrinsics']['coeffs']):
                    raise ValueError('Live distortion differs from calibration')
        except BaseException:
            self.close(); raise
        return self

    def read(self, qpos, device='cpu'):
        qpos = np.asarray(qpos, np.float32)
        if qpos.shape != (4,) or not np.isfinite(qpos).all(): raise ValueError('Expected four finite joint positions')
        packet = self.source.read()
        cloud = self.builder(packet['depth_frames'], packet['depth_scales'], packet['frames'])
        self.history.append((cloud, qpos.copy()))
        while len(self.history) < self.history.maxlen: self.history.appendleft(self.history[0])
        return {key:torch.from_numpy(np.stack(values))[None].to(device) for key, values in
                [('point_cloud', [p for p,q in self.history]), ('agent_pos', [q for p,q in self.history])]}

    def close(self): self.source.close()
