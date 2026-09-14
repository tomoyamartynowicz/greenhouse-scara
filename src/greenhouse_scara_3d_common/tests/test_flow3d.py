import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from test_3d import config, build_policy
from greenhouse_scara_3d_common.vendor.dp3.model.common.normalizer import SingleFieldLinearNormalizer


class TimeVelocity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale, self.times = scale, []

    def forward(self, sample, timestep, **kwargs):
        self.times.append(timestep.detach().clone())
        return (2 * timestep[:, None, None] / self.scale).expand_as(sample)


class OnesVelocity(nn.Module):
    def forward(self, sample, timestep, **kwargs):
        return torch.ones_like(sample)


class Flow3DTests(unittest.TestCase):
    def test_same_encoder_and_unet_parameter_shapes_as_dp3(self):
        with tempfile.TemporaryDirectory() as tmp:
            dp, fm = [build_policy(config(k, tmp)) for k in ('dp3', 'flow3d')]
            shapes = lambda module: {k: tuple(v.shape) for k, v in module.named_parameters()}
            self.assertEqual(shapes(dp), shapes(fm))

    def test_solvers_integrate_known_velocity_and_count_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = build_policy(config('flow3d', tmp))
            policy.num_inference_steps = 4
            data = torch.zeros(2, 8, 4)
            mask = torch.zeros_like(data, dtype=torch.bool)
            mask[:, 0] = True
            source = torch.randn(data.shape, generator=torch.Generator().manual_seed(5))
            for method, integral, count in [('euler', .75, 4), ('heun', 1., 8)]:
                policy.integration_method = method
                policy.model = TimeVelocity(policy.time_scale)
                result = policy.conditional_sample(data, mask, generator=torch.Generator().manual_seed(5))
                expected = torch.where(mask, data, source + integral)
                torch.testing.assert_close(result, expected)
                self.assertEqual(len(policy.model.times), count)
                self.assertTrue(all(t.is_floating_point() for t in policy.model.times))

    def test_loss_regresses_action_minus_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = build_policy(config('flow3d', tmp))
            for key in ('point_cloud', 'agent_pos', 'action'):
                policy.normalizer[key] = SingleFieldLinearNormalizer.create_identity()
            policy.model = OnesVelocity()
            batch = {'obs': {'point_cloud': torch.zeros(2, 2, 16, 3),
                             'agent_pos': torch.zeros(2, 2, 4)}, 'action': torch.ones(2, 8, 4)}
            with patch('greenhouse_scara_3d_flow_matching_policy.policy.torch.randn_like',
                       side_effect=torch.zeros_like):
                loss, _ = policy.compute_loss(batch)
            self.assertEqual(loss.item(), 0.)


if __name__ == '__main__':
    unittest.main()
