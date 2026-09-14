"""Existing SCARA rectified-flow objective/U-Net with the upstream DP3 encoder."""
from omegaconf import OmegaConf
from greenhouse_scara_3d_common.vendor.dp3.model.vision.pointnet_extractor import DP3Encoder
from .core.policy.flow_matching_unet_image_policy import FlowMatchingUnetImagePolicy


class Encoder(DP3Encoder):
    def output_shape(self):
        return (super().output_shape(),)  # Existing flow policy expects a one-element tuple.


def build(cfg, shape_meta):
    options = dict(cfg['policy'])
    encoder_dim = options.pop('encoder_output_dim')
    encoder = Encoder(observation_space={key: spec['shape'] for key, spec in shape_meta['obs'].items()},
                      out_channel=encoder_dim, use_pc_color=cfg['pointcloud']['use_color'],
                      pointcloud_encoder_cfg=OmegaConf.create(dict(in_channels=3, out_channels=encoder_dim,
                                              use_layernorm=True, final_norm='layernorm')))
    return FlowMatchingUnetImagePolicy(shape_meta=shape_meta, obs_encoder=encoder,
        horizon=cfg['horizon'], n_obs_steps=cfg['n_obs_steps'], n_action_steps=cfg['n_action_steps'], **options)
