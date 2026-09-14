from typing import Dict
import torch
import torch.nn.functional as F
from einops import reduce

from greenhouse_scara_3d_common.vendor.dp3.model.common.normalizer import LinearNormalizer
from greenhouse_scara_3d_flow_matching_policy.core.policy.base_image_policy import BaseImagePolicy
from greenhouse_scara_3d_flow_matching_policy.core.model.diffusion.conditional_unet1d import ConditionalUnet1D
from greenhouse_scara_3d_flow_matching_policy.core.model.diffusion.mask_generator import LowdimMaskGenerator
from torch.nn import Module as MultiImageObsEncoder
from greenhouse_scara_3d_common.vendor.dp3.common.pytorch_util import dict_apply

class FlowMatchingUnetImagePolicy(BaseImagePolicy):
    """Image-conditioned rectified-flow policy for action trajectories.

    The network architecture and observation path intentionally match the SCARA
    diffusion baseline. Only the generative objective and sampler differ:
    training regresses the constant velocity from Gaussian noise to a normalized
    expert trajectory, and inference integrates that learned velocity field.
    """

    def __init__(self, 
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=16,
            integration_method='heun',
            time_scale=100.0,
            noise_scheduler=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected flow policy arguments: {sorted(kwargs)}")

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # Reuse the temporal conditional U-Net architecture; its output now
        # represents velocity rather than a diffusion noise residual.
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = int(num_inference_steps)
        self.integration_method = str(integration_method).lower()
        self.time_scale = float(time_scale)
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if self.integration_method not in {'euler', 'heun'}:
            raise ValueError("integration_method must be 'euler' or 'heun'")
        if self.time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if noise_scheduler is not None:
            raise ValueError(
                "FlowMatchingUnetImagePolicy does not use a diffusion noise scheduler"
            )
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            ):
        model = self.model

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # Integrate dx/dt = v_theta(x, t, observation) from noise (t=0)
        # to the action distribution (t=1). Passing a float tensor is important:
        # ConditionalUnet1D otherwise converts Python scalars to integer timesteps.
        time_grid = torch.linspace(
            0.0,
            1.0,
            self.num_inference_steps + 1,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        batch_size = trajectory.shape[0]
        for index in range(self.num_inference_steps):
            t0 = time_grid[index]
            t1 = time_grid[index + 1]
            dt = t1 - t0
            trajectory[condition_mask] = condition_data[condition_mask]

            model_time = (t0 * self.time_scale).expand(batch_size)
            velocity = model(trajectory, model_time,
                local_cond=local_cond, global_cond=global_cond)

            if self.integration_method == 'euler':
                trajectory = trajectory + dt * velocity
            else:
                proposal = trajectory + dt * velocity
                proposal[condition_mask] = condition_data[condition_mask]
                next_model_time = (t1 * self.time_scale).expand(batch_size)
                next_velocity = model(
                    proposal,
                    next_model_time,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
                trajectory = trajectory + 0.5 * dt * (velocity + next_velocity)
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Rectified conditional flow matching. x0 is Gaussian source noise and
        # x1 is the normalized expert trajectory. Along the straight probability
        # path x_t=(1-t)x0+t*x1, the target velocity is constant: x1-x0.
        source = torch.randn_like(trajectory)
        bsz = trajectory.shape[0]
        flow_time = torch.rand(
            (bsz,), device=trajectory.device, dtype=trajectory.dtype
        )
        broadcast_time = flow_time.reshape(
            bsz, *([1] * (trajectory.ndim - 1))
        )
        interpolated = (
            (1.0 - broadcast_time) * source + broadcast_time * trajectory
        )
        target = trajectory - source
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        interpolated[condition_mask] = cond_data[condition_mask]

        # Scale continuous t to the range used by the inherited sinusoidal
        # embedding (the diffusion baseline used approximately 0..100).
        model_time = flow_time * self.time_scale
        pred = self.model(interpolated, model_time,
            local_cond=local_cond, global_cond=global_cond)

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
