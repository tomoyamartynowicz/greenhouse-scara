"""Raw RealSense observations to the same calibrated point-cloud input as the loader."""
from collections import deque
import numpy as np
import torch
import warnings
from greenhouse_scara_common.live import MultiCameraSource
from .pointcloud import PointCloudBuilder


class CalibratedCameraSource:
    """Raw packets plus verification against saved training geometry."""
    def __init__(self, cfg, fps=30):
        self.cfg = cfg
        self.builder = PointCloudBuilder(cfg['calibration'], cfg['pointcloud'])
        names = cfg['pointcloud']['cameras']
        cams = cfg['calibration']['cameras']
        sizes = {(cams[n]['width'], cams[n]['height']) for n in names}
        if len(sizes) != 1: raise ValueError('Live cameras currently require the same capture resolution')
        specs = {n+'_depth': dict(camera=n, type='depth', aligned_to_rgb=cams[n]['aligned_to_rgb']) for n in names}
        self.source = MultiCameraSource(specs, serials={n:cams[n]['serial'] for n in names}, size=next(iter(sizes)), fps=fps)

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
                if (actual['width'], actual['height']) != (saved['width'], saved['height']):
                    raise ValueError('Live image profile differs from calibration')
                matches = (all(np.isclose(actual['intrinsics'][key], saved['intrinsics'][key], rtol=1e-5)
                               for key in ('fx', 'fy', 'cx', 'cy'))
                           and actual['intrinsics']['distortion_model'] == saved['intrinsics']['distortion_model']
                           and np.allclose(actual['intrinsics']['coeffs'], saved['intrinsics']['coeffs']))
                if not matches:
                    provenance = self.cfg['calibration'].get('provenance', {})
                    if not (provenance.get('approximate_intrinsics') and provenance.get('user_accepted_approximation')):
                        raise ValueError(f'{source.color_name}: live intrinsics differ from training calibration')
                    warnings.warn('Using the same approximate intrinsics as training, borrowed from another camera; metric geometry is approximate', stacklevel=2)
        except BaseException:
            self.close(); raise
        return self

    def read(self):
        return self.source.read()

    def close(self):
        self.source.close()


class LivePointCloudSource(CalibratedCameraSource):
    """Legacy observation-only interface; deployment uses raw packets above."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.history = deque(maxlen=cfg['n_obs_steps'])

    def read(self, qpos, device='cpu'):
        qpos = np.asarray(qpos, np.float32)
        dimension = 5 if self.cfg.get('include_scissor_roll', False) else 4
        if qpos.shape != (dimension,) or not np.isfinite(qpos).all(): raise ValueError(f'Expected {dimension} finite state values')
        packet = self.source.read()
        cloud = self.builder(packet['depth_frames'], packet['depth_scales'], packet['frames'])
        self.history.append((cloud, qpos.copy()))
        while len(self.history) < self.history.maxlen: self.history.appendleft(self.history[0])
        return {key:torch.from_numpy(np.stack(values))[None].to(device) for key, values in
                [('point_cloud', [p for p,q in self.history]), ('agent_pos', [q for p,q in self.history])]}

    def close(self): self.source.close()
