"""SCARA configuration of the vendored upstream DP3 policy."""
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from greenhouse_scara_3d_common.vendor.dp3.policy.dp3 import DP3


def build(cfg, shape_meta):
    options = dict(cfg['policy'])
    encoder_dim = options.pop('encoder_output_dim')
    noise = options.pop('noise_scheduler')
    encoder = OmegaConf.create(dict(in_channels=3, out_channels=encoder_dim,
                                    use_layernorm=True, final_norm='layernorm'))
    return DP3(shape_meta=shape_meta, horizon=cfg['horizon'], n_obs_steps=cfg['n_obs_steps'],
               n_action_steps=cfg['n_action_steps'], noise_scheduler=DDIMScheduler(**noise),
               encoder_output_dim=encoder_dim, pointcloud_encoder_cfg=encoder,
               use_pc_color=cfg['pointcloud']['use_color'], pointnet_type='pointnet', **options)
