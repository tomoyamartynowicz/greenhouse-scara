"""ACT CVAE/transformer with a pooled DP3 PointNet feature instead of RGB CNNs."""
from types import SimpleNamespace
import torch
from torch import nn
from greenhouse_scara_act.greenhouse_act_detr.models.detr_vae import DETRVAE, build_encoder
from greenhouse_scara_act.greenhouse_act_detr.models.transformer import build_transformer
from greenhouse_scara_3d_common.vendor.dp3.model.vision.pointnet_extractor import PointNetEncoderXYZ, PointNetEncoderXYZRGB
from greenhouse_scara_3d_common.vendor.dp3.model.common.normalizer import LinearNormalizer


class PointBackbone(nn.Module):
    def __init__(self, channels, feature_dim, hidden_dim):
        super().__init__()
        cls = PointNetEncoderXYZRGB if channels == 6 else PointNetEncoderXYZ
        self.encoder = cls(in_channels=channels, out_channels=feature_dim,
                           use_layernorm=True, final_norm='layernorm')
        self.num_channels = feature_dim
        self.hidden_dim = hidden_dim

    def forward(self, points):
        features = self.encoder(points)[..., None, None]
        # One pooled visual token; no artificial spatial ordering of FPS points.
        # ACT's transformer broadcasts the positional tensor over the batch.
        position = features.new_zeros((1, self.hidden_dim, 1, 1))
        return [features], [position]


class ACT3D(nn.Module):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        if cfg['n_obs_steps'] != 1:
            raise ValueError('ACT3D uses one current pointcloud/state observation')
        options = dict(cfg['policy'])
        self.kl_weight = options.pop('kl_weight')
        feature_dim = options.pop('encoder_output_dim')
        args = SimpleNamespace(**options)
        backbone = PointBackbone(shape_meta.obs.point_cloud.shape[-1], feature_dim, args.hidden_dim)
        self.model = DETRVAE([backbone], build_transformer(args), build_encoder(args),
                             shape_meta.action.shape[0], cfg['horizon'], ['point_cloud'], [0])
        self.normalizer = LinearNormalizer()
        self.n_action_steps = cfg['n_action_steps']

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        obs = self.normalizer.normalize(batch['obs'])
        actions = self.normalizer['action'].normalize(batch['action'])
        pad = batch['is_pad']
        prediction, _, (mu, logvar) = self.model(
            obs['agent_pos'][:, -1], {'point_cloud': obs['point_cloud'][:, -1]}, None, actions, pad)
        valid = (~pad).unsqueeze(-1)
        l1 = ((prediction - actions).abs() * valid).sum() / (valid.sum() * actions.shape[-1]).clamp_min(1)
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return l1 + self.kl_weight * kl, {'l1': l1.detach(), 'kl': kl.detach()}

    def predict_action(self, observations):
        obs = self.normalizer.normalize(observations)
        prediction, _, _ = self.model(obs['agent_pos'][:, -1],
                                     {'point_cloud': obs['point_cloud'][:, -1]}, None)
        actions = self.normalizer['action'].unnormalize(prediction)
        return {'action': actions[:, :self.n_action_steps], 'action_pred': actions}


def build(cfg, shape_meta):
    return ACT3D(cfg, shape_meta)
