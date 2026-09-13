import unittest

import torch
import torch.nn as nn

from greenhouse_flow_matching_policy.model.common.normalizer import (
    SingleFieldLinearNormalizer,
)
from greenhouse_flow_matching_policy.policy.flow_matching_unet_image_policy import (
    FlowMatchingUnetImagePolicy,
)


class TinyObsEncoder(nn.Module):
    def __init__(self, output_dim: int = 8):
        super().__init__()
        self.projection = nn.Linear(5, output_dim)
        self.output_dim = output_dim

    def output_shape(self):
        return (self.output_dim,)

    def forward(self, observation):
        image_mean = observation["camera"].mean(
            dim=(-3, -2, -1), keepdim=False
        ).unsqueeze(-1)
        features = torch.cat([image_mean, observation["qpos"]], dim=-1)
        return self.projection(features)


def policy_kwargs(integration_method: str = "heun") -> dict:
    return {
        "shape_meta": {
            "obs": {
                "camera": {"shape": [3, 8, 8], "type": "rgb"},
                "qpos": {"shape": [4], "type": "low_dim"},
            },
            "action": {"shape": [4]},
        },
        "obs_encoder": TinyObsEncoder(),
        "horizon": 8,
        "n_action_steps": 4,
        "n_obs_steps": 2,
        "num_inference_steps": 3,
        "integration_method": integration_method,
        "time_scale": 100.0,
        "obs_as_global_cond": True,
        "diffusion_step_embed_dim": 16,
        "down_dims": (16, 32),
        "kernel_size": 3,
        "n_groups": 4,
        "cond_predict_scale": True,
    }


def make_policy(integration_method: str) -> FlowMatchingUnetImagePolicy:
    policy = FlowMatchingUnetImagePolicy(**policy_kwargs(integration_method))
    for key in ("camera", "qpos", "action"):
        policy.normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    return policy


def make_batch():
    return {
        "obs": {
            "camera": torch.rand(2, 2, 3, 8, 8),
            "qpos": torch.rand(2, 2, 4),
        },
        "action": torch.rand(2, 8, 4),
    }


class FlowMatchingPolicyTest(unittest.TestCase):
    def test_flow_loss_is_finite_and_backpropagates(self):
        policy = make_policy("heun")
        loss = policy.compute_loss(make_batch())

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in policy.model.parameters()
            )
        )

    def test_flow_sampling_returns_expected_action_shape(self):
        for integration_method in ("euler", "heun"):
            with self.subTest(integration_method=integration_method):
                policy = make_policy(integration_method).eval()
                with torch.inference_mode():
                    result = policy.predict_action(make_batch()["obs"])

                self.assertEqual(result["action"].shape, (2, 4, 4))
                self.assertEqual(result["action_pred"].shape, (2, 8, 4))
                self.assertTrue(torch.isfinite(result["action"]).all())

    def test_diffusion_scheduler_is_rejected(self):
        kwargs = policy_kwargs("euler")
        kwargs["noise_scheduler"] = object()
        with self.assertRaisesRegex(
            ValueError, "does not use a diffusion noise scheduler"
        ):
            FlowMatchingUnetImagePolicy(**kwargs)


if __name__ == "__main__":
    unittest.main()
