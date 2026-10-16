"""Load 2D ACT/DP/Flow checkpoints without constructing training workspaces."""
from __future__ import annotations
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from .data import image_specs
from .live import observation_from_history

ROOT = Path(__file__).resolve().parents[1]


def check_variant(meta, modality):
    expected = {'camera_top_rgb', 'camera_bottom_rgb'}
    if modality == 'rgbd':
        expected |= {'camera_top_depth', 'camera_bottom_depth'}
    if set(meta['obs']) - {'qpos'} != expected:
        raise ValueError('Checkpoint cameras/modality do not match the selected variant')
    if list(meta['obs']['qpos']['shape']) != [5] or list(meta['action']['shape']) != [5]:
        raise ValueError('This runner requires [J1, J2, J3, J4, scissor_roll] in state and actions')
    return image_specs(meta)


class Predictor:
    def __init__(self, model, modality, checkpoint, device='cpu', inference_steps=None):
        self.model_kind = model
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu')
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if model == 'act':
            sys.path.insert(0, str(ROOT / 'greenhouse_scara_act'))
            from policy import ACTPolicy
            with (self.checkpoint.parent / 'config.pkl').open('rb') as f:
                self.cfg = pickle.load(f)
            with (self.checkpoint.parent / 'dataset_stats.pkl').open('rb') as f:
                self.stats = pickle.load(f)
            self.specs = check_variant(self.cfg['inputs_config']['shape_meta'], modality)
            if not self.cfg['inputs_config'].get('include_scissor_roll'):
                raise ValueError('ACT checkpoint does not include scissor roll')
            state = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            state = state.get('policy', state)
            config = dict(self.cfg['policy_config'])
            config['num_queries'] = state['model.query_embed.weight'].shape[0]
            config['pretrained_backbone'] = False
            self.policy = ACTPolicy(config)
            self.policy.load_state_dict(state)
            self.n_obs_steps = 1
            for name in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std'):
                value = np.asarray(self.stats[name])
                if value.shape != (5,) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid ACT normalization: {name}')
                if name.endswith('std') and np.any(value <= 0):
                    raise ValueError(f'Nonpositive ACT normalization: {name}')
        else:
            import dill
            import hydra
            from omegaconf import OmegaConf
            kind = {'dp': 'diffusion', 'fm': 'flow_matching'}[model]
            sys.path.insert(0, str(ROOT / f'greenhouse_scara_{kind}_policy'))
            OmegaConf.register_new_resolver('eval', eval, replace=True)
            payload = torch.load(self.checkpoint, map_location='cpu', pickle_module=dill, weights_only=False)
            self.cfg = payload['cfg']
            if not str(self.cfg.policy._target_).startswith(f'greenhouse_{kind}_policy.'):
                raise ValueError('Checkpoint model family does not match the requested model')
            self.specs = check_variant(self.cfg.task.shape_meta, modality)
            if not self.cfg.task.get('include_scissor_roll', False):
                raise ValueError('Checkpoint does not include scissor roll')
            config = OmegaConf.create(OmegaConf.to_container(self.cfg.policy, resolve=True))
            if 'pretrained' in config.obs_encoder:
                config.obs_encoder.pretrained = False
            self.policy = hydra.utils.instantiate(config)
            key = 'ema_model' if self.cfg.training.use_ema else 'model'
            self.policy.load_state_dict(payload['state_dicts'][key])
            self.n_obs_steps = int(self.cfg.n_obs_steps)
            self.policy.reset()
            if inference_steps is not None:
                if inference_steps < 1:
                    raise ValueError('Inference steps must be positive')
                self.policy.num_inference_steps = inference_steps
            del payload
        self.policy.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, history):
        if len(history) != self.n_obs_steps:
            raise ValueError(f'Expected {self.n_obs_steps} observations, got {len(history)}')
        obs = observation_from_history(history, self.specs, self.device)
        if self.model_kind == 'act':
            qpos = (obs['qpos'][:, -1] - torch.as_tensor(self.stats['qpos_mean'], device=self.device))
            qpos = qpos / torch.as_tensor(self.stats['qpos_std'], device=self.device)
            images = {key: obs[key][:, -1] for key in self.specs}
            actions = self.policy(qpos.float(), images)[0].cpu().numpy()
            actions = actions * self.stats['action_std'] + self.stats['action_mean']
        else:
            actions = self.policy.predict_action(obs)['action'][0].cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 5 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid predicted actions: {actions.shape}; expected finite (T,5)')
        return actions
