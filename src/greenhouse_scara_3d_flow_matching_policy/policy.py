"""Conditional rectified flow with exactly DP3's PointNet/U-Net and action interface.

Ordinary velocity regression, not the consistency-trained FlowPolicy method.
"""
import math
import torch
from omegaconf import OmegaConf
from greenhouse_scara_3d_common.vendor.dp3.policy.dp3 import DP3


class FlowMatchingDP3(DP3):
    def __init__(self, *, integration_method='heun', time_scale=100.,
                 num_inference_steps=8, **kwargs):
        if integration_method not in ('euler', 'heun'):
            raise ValueError('integration_method must be euler or heun')
        if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
            raise ValueError('num_inference_steps must be a positive integer')
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('time_scale must be positive and finite')
        if not kwargs.get('obs_as_global_cond', True):
            raise ValueError('This flow baseline supports global observation conditioning only')
        super().__init__(noise_scheduler=None, num_inference_steps=num_inference_steps, **kwargs)
        self.integration_method = integration_method
        self.time_scale = time_scale

    def conditional_sample(self, condition_data, condition_mask,
                           condition_data_pc=None, condition_mask_pc=None,
                           local_cond=None, global_cond=None, generator=None, **kwargs):
        trajectory = torch.randn(condition_data.shape, device=condition_data.device,
                                 dtype=condition_data.dtype, generator=generator)
        grid = torch.linspace(0., 1., self.num_inference_steps + 1,
                              device=trajectory.device, dtype=trajectory.dtype)
        for t0, t1 in zip(grid[:-1], grid[1:]):
            trajectory = torch.where(condition_mask, condition_data, trajectory)
            velocity = self.model(trajectory, (t0 * self.time_scale).expand(len(trajectory)),
                                  local_cond=local_cond, global_cond=global_cond)
            proposal = trajectory + (t1 - t0) * velocity
            if self.integration_method == 'heun':
                proposal = torch.where(condition_mask, condition_data, proposal)
                correction = self.model(proposal, (t1 * self.time_scale).expand(len(trajectory)),
                                        local_cond=local_cond, global_cond=global_cond)
                trajectory = trajectory + .5 * (t1 - t0) * (velocity + correction)
            else:
                trajectory = proposal
        return torch.where(condition_mask, condition_data, trajectory)

    def compute_loss(self, batch):
        obs = self.normalizer.normalize(batch['obs'])
        actions = self.normalizer['action'].normalize(batch['action'])
        if not self.use_pc_color:
            obs['point_cloud'] = obs['point_cloud'][..., :3]
        features = self.obs_encoder({key: value[:, :self.n_obs_steps].reshape(-1, *value.shape[2:])
                                     for key, value in obs.items()})
        shape = (len(actions), self.n_obs_steps, -1) if 'cross_attention' in self.condition_type else (len(actions), -1)
        global_cond = features.reshape(shape)
        source = torch.randn_like(actions)
        t = torch.rand(len(actions), device=actions.device, dtype=actions.dtype)
        interpolated = (1. - t[:, None, None]) * source + t[:, None, None] * actions
        velocity = self.model(interpolated, t * self.time_scale, global_cond=global_cond)
        # DP3's dataset repeats boundary actions; keep that same loss convention.
        loss = (velocity - (actions - source)).square().mean()
        return loss, {'bc_loss': loss.detach().item()}


def build(cfg, shape_meta):
    options = dict(cfg['policy'])
    encoder_dim = options.pop('encoder_output_dim')
    encoder = OmegaConf.create(dict(in_channels=3, out_channels=encoder_dim,
                                   use_layernorm=True, final_norm='layernorm'))
    return FlowMatchingDP3(shape_meta=shape_meta, horizon=cfg['horizon'],
        n_obs_steps=cfg['n_obs_steps'], n_action_steps=cfg['n_action_steps'],
        encoder_output_dim=encoder_dim, pointcloud_encoder_cfg=encoder,
        use_pc_color=cfg['pointcloud']['use_color'], pointnet_type='pointnet', **options)
