"""Round-trip real small ACT/DP/Flow checkpoints through the deployment loader."""
import gc
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for path in ('src', 'src/greenhouse_scara_act', 'src/greenhouse_scara_diffusion_policy', 'src/greenhouse_scara_flow_matching_policy'):
    sys.path.insert(0, str(ROOT / path))
from greenhouse_scara_common.inference import Predictor, check_variant
from policy import ACTPolicy


def meta_for(modality):
    obs = {f'{name}_rgb': dict(camera=name, type='rgb', shape=[3, 32, 32])
           for name in ('camera_top', 'camera_bottom')}
    if modality == 'rgbd':
        obs.update({f'{name}_depth': dict(camera=name, type='depth', shape=[2, 32, 32],
                                       depth_normalization_m=1., aligned_to_rgb=True)
                    for name in ('camera_top', 'camera_bottom')})
    obs['qpos'] = dict(type='low_dim', shape=[5])
    return dict(obs=obs, action=dict(shape=[5]))


def make_checkpoint(folder, kind, modality):
    meta = meta_for(modality)
    if kind == 'act':
        specs = {key: spec for key, spec in meta['obs'].items() if key != 'qpos'}
        config = dict(stream_specs=specs, camera_names=list(specs), state_dim=5,
                      hidden_dim=32, dim_feedforward=64, nheads=4, enc_layers=1, dec_layers=1,
                      num_queries=4, kl_weight=1, pretrained_backbone=False)
        policy = ACTPolicy(config)
        torch.save(policy.state_dict(), folder / 'policy_last.ckpt')
        with (folder / 'config.pkl').open('wb') as f:
            pickle.dump(dict(policy_config=config, inputs_config=dict(shape_meta=meta,
                            include_scissor_roll=True)), f)
        with (folder / 'dataset_stats.pkl').open('wb') as f:
            pickle.dump(dict(qpos_mean=np.zeros(5,dtype=np.float32), qpos_std=np.ones(5,dtype=np.float32),
                             action_mean=np.zeros(5,dtype=np.float32), action_std=np.ones(5,dtype=np.float32)), f)
        return folder / 'policy_last.ckpt'
    import dill
    family = 'diffusion' if kind == 'dp' else 'flow_matching'
    classname = 'DiffusionUnetImagePolicy' if kind == 'dp' else 'FlowMatchingUnetImagePolicy'
    config = dict(_target_=f'greenhouse_{family}_policy.policy.{family}_unet_image_policy.{classname}',
                  shape_meta=meta, obs_encoder=dict(_target_='greenhouse_scara_common.vision.MultiModalObsEncoder',
                  shape_meta=meta, pretrained=False), horizon=8, n_obs_steps=2, n_action_steps=4,
                  num_inference_steps=1, down_dims=[16,32], diffusion_step_embed_dim=16, n_groups=4)
    if kind == 'dp':
        config['noise_scheduler'] = dict(_target_='diffusers.DDIMScheduler', num_train_timesteps=10)
        from greenhouse_diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
    else:
        from greenhouse_flow_matching_policy.model.common.normalizer import SingleFieldLinearNormalizer
    policy = hydra.utils.instantiate(OmegaConf.create(config))
    for key in (*meta['obs'], 'action'):
        policy.normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    cfg = OmegaConf.create(dict(policy=config, task=dict(shape_meta=meta, include_scissor_roll=True),
                                training=dict(use_ema=True), n_obs_steps=2))
    # Only EMA is present, so accidentally using raw model weights will fail.
    torch.save(dict(cfg=cfg, state_dicts=dict(ema_model=policy.state_dict())), folder / 'latest.ckpt', pickle_module=dill)
    return folder / 'latest.ckpt'


class InferenceTests(unittest.TestCase):
    def test_all_six_checkpoint_loads_and_live_predictions(self):
        torch.set_num_threads(1)
        for kind in ('act', 'dp', 'fm'):
            for modality in ('rgb', 'rgbd'):
                with self.subTest(kind=kind, modality=modality), tempfile.TemporaryDirectory() as temp:
                    checkpoint = make_checkpoint(Path(temp), kind, modality)
                    predictor = Predictor(kind, modality, checkpoint, 'cpu', 1)
                    sample = dict(frames={name: np.zeros((48,64,3),dtype=np.uint8)
                                          for name in ('camera_top','camera_bottom')},
                                  depth_frames={name: np.ones((48,64),dtype=np.uint16)*300
                                                for name in ('camera_top','camera_bottom')},
                                  depth_scales={'camera_top': .001, 'camera_bottom': .001})
                    history = [(sample, np.array([.5,0,1,0,-1],dtype=np.float32))]*predictor.n_obs_steps
                    actions = predictor.predict(history)
                    self.assertEqual(actions.shape,(4,5))
                    self.assertTrue(np.isfinite(actions).all())
                    with self.assertRaises(ValueError):
                        Predictor(kind, 'rgbd' if modality=='rgb' else 'rgb', checkpoint, 'cpu', 1)
                    del predictor
                    gc.collect()

    def test_four_joint_checkpoint_rejected(self):
        meta = meta_for('rgb')
        meta['action']['shape'] = [4]
        with self.assertRaises(ValueError):
            check_variant(meta, 'rgb')


if __name__ == '__main__':
    unittest.main()
