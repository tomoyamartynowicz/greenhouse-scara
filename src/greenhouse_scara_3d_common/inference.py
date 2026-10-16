"""Five-action 3D deployment adapter using the checkpoint's exact geometry."""
import numpy as np
import torch
from .runtime import load_checkpoint
from .pointcloud import PointCloudBuilder


class Predictor3D:
    def __init__(self, model, checkpoint, device='auto', inference_steps=None):
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        from pathlib import Path
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.policy, state = load_checkpoint(self.checkpoint, self.device)
        self.cfg = state['config']
        expected = {'act3d': 'act3d', 'dp3': 'dp3', 'fm3': 'flow3d'}[model]
        if self.cfg['model'] != expected or not self.cfg.get('include_scissor_roll', False):
            raise ValueError('Checkpoint must match the selected 3D model and include scissor roll')
        self.model_kind = model
        self.n_obs_steps = self.cfg['n_obs_steps']
        self.builder = PointCloudBuilder(self.cfg['calibration'], self.cfg['pointcloud'])
        self.specs = {}
        for name in self.cfg['pointcloud']['cameras']:
            camera = self.cfg['calibration']['cameras'][name]
            self.specs[name + '_depth'] = dict(camera=name, type='depth', shape=[2, camera['height'], camera['width']],
                                              aligned_to_rgb=camera['aligned_to_rgb'])
            if self.cfg['pointcloud']['use_color']:
                self.specs[name + '_rgb'] = dict(camera=name, type='rgb', shape=[3, camera['height'], camera['width']])
        if inference_steps is not None:
            if model == 'act3d' or inference_steps < 1:
                raise ValueError('Inference steps must be positive and apply only to DP3/Flow3D')
            self.policy.num_inference_steps = inference_steps

    def camera_source(self, fps=30):
        from .live import CalibratedCameraSource
        return CalibratedCameraSource(self.cfg, fps=fps)

    @torch.inference_mode()
    def predict(self, history):
        if len(history) != self.n_obs_steps:
            raise ValueError(f'Expected {self.n_obs_steps} observations')
        clouds, states = [], []
        for sample, qpos in history:
            qpos = np.asarray(qpos, dtype=np.float32)
            if qpos.shape != (5,) or not np.isfinite(qpos).all():
                raise ValueError('Expected five finite state values including scissor roll')
            for name in self.cfg['pointcloud']['cameras']:
                if name in sample.get('depth_aligned', {}) and sample['depth_aligned'][name] != self.cfg['calibration']['cameras'][name]['aligned_to_rgb']:
                    raise ValueError('Live depth alignment differs from checkpoint')
            clouds.append(self.builder(sample['depth_frames'], sample['depth_scales'], sample.get('frames')))
            states.append(qpos)
        obs = dict(point_cloud=torch.from_numpy(np.stack(clouds))[None].to(self.device),
                   agent_pos=torch.from_numpy(np.stack(states))[None].to(self.device))
        output = self.policy.predict_action(obs)
        actions = output['action_pred' if self.model_kind == 'act3d' else 'action'][0].cpu().numpy()
        if actions.ndim != 2 or actions.shape[1] != 5 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError('Invalid predicted five-dimensional action chunk')
        return actions
