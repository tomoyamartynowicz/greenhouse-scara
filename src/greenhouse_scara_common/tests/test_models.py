"""Exercise actual policy losses and inference with two RGB-D cameras."""
import unittest

from test_inputs import shape_meta, MultiModalObsEncoder
import torch
from diffusers import DDIMScheduler
from policy import ACTPolicy
from greenhouse_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from greenhouse_diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer as DPNorm
from greenhouse_flow_matching_policy.policy.flow_matching_unet_image_policy import FlowMatchingUnetImagePolicy
from greenhouse_flow_matching_policy.model.common.normalizer import SingleFieldLinearNormalizer as FMNorm


class ModelTests(unittest.TestCase):
    def test_act_two_rgbd_cameras(self):
        meta = shape_meta()
        specs = {key: spec for key, spec in meta['obs'].items() if key != 'qpos'}
        policy = ACTPolicy(dict(stream_specs=specs, camera_names=list(specs),
            state_dim=4, hidden_dim=32, dim_feedforward=64, nheads=4,
            enc_layers=1, dec_layers=1, num_queries=4, kl_weight=1,
            pretrained_backbone=False))
        self.assertEqual(len(policy.model.backbones), 2)
        self.assertEqual(policy.model.backbone_indices, [0, 1, 0, 1])
        images = {key: torch.rand(2, *spec['shape'], requires_grad=True) for key, spec in specs.items()}
        qpos = torch.rand(2, 4)
        loss = policy(qpos, images, torch.rand(2, 4, 4), torch.zeros(2, 4, dtype=torch.bool))['loss']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for image in images.values():
            self.assertIsNotNone(image.grad)
            self.assertGreater(torch.count_nonzero(image.grad), 0)
        for backbone in policy.model.backbones:
            gradients = [p.grad for p in backbone.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None and torch.count_nonzero(g) for g in gradients))
        policy.eval()
        with torch.no_grad():
            prediction = policy(qpos, images)
        self.assertEqual(tuple(prediction.shape), (2, 4, 4))
        self.assertTrue(torch.isfinite(prediction).all())

    def _check_unet(self, cls, norm, **kwargs):
        meta = shape_meta()
        policy = cls(shape_meta=meta, obs_encoder=MultiModalObsEncoder(meta),
            horizon=8, n_obs_steps=2, n_action_steps=4, num_inference_steps=1,
            down_dims=(16, 32), diffusion_step_embed_dim=16, n_groups=4, **kwargs)
        for key in (*meta['obs'], 'action'):
            policy.normalizer[key] = norm.create_identity()
        obs = {key: torch.rand(2, 2, *spec['shape']) for key, spec in meta['obs'].items()}
        loss = policy.compute_loss({'obs': obs, 'action': torch.rand(2, 8, 4)})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for encoder in policy.obs_encoder.models.values():
            gradient = encoder.conv1.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(torch.count_nonzero(gradient), 0)
        policy.eval()
        with torch.no_grad():
            prediction = policy.predict_action(obs)['action']
        self.assertEqual(tuple(prediction.shape), (2, 4, 4))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_diffusion_two_rgbd_cameras(self):
        self._check_unet(DiffusionUnetImagePolicy, DPNorm,
                         noise_scheduler=DDIMScheduler(num_train_timesteps=100))

    def test_flow_matching_two_rgbd_cameras(self):
        self._check_unet(FlowMatchingUnetImagePolicy, FMNorm)


if __name__ == '__main__':
    unittest.main()
