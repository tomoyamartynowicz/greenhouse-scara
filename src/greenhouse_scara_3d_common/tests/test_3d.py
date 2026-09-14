from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import h5py
import numpy as np
import torch
import yaml

SRC=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SRC))
from greenhouse_scara_3d_common.pointcloud import PointCloudBuilder, camera_points, validate_calibration
from greenhouse_scara_3d_common.dataset import PointCloudDataset, split_paths
from greenhouse_scara_3d_common.runtime import MODELS, build_policy, train, load_checkpoint

# This repository currently bundles DP3; other local 3D projects are optional.
AVAILABLE_MODELS = {k: v for k, v in MODELS.items() if (SRC / v).is_dir()}


def settings():
    return dict(cameras=['camera_top'],reference_frame='camera_top',num_points=16,
                use_color=False,min_depth_m=.02,max_depth_m=1.5,pixel_stride=1,
                fps_candidates=64,crop_min=None,crop_max=None)


def calibration():
    return {'cameras': {'camera_top':dict(serial='test-only',width=8,height=8,aligned_to_rgb=True,
            intrinsics=dict(fx=8.,fy=8.,cx=3.5,cy=3.5,distortion_model='none',coeffs=[0.]*5),to_reference=None)}}


def config(model, tmp):
    cfg=yaml.safe_load((SRC/MODELS[model]/'config.yaml').read_text())
    cfg.update(model=model,pointcloud=settings(),calibration=calibration(),dataset_dir=str(tmp),
               output_dir=str(Path(tmp)/('run_'+model)),device='cpu',epochs=1,batch_size=4,val_batch_size=4,
               num_workers=0,horizon=8,n_action_steps=4,max_train_steps=1,max_val_steps=1,warmup_steps=0)
    if model in ('dp3','flow3d'):cfg['policy'].update(down_dims=[16,32],diffusion_step_embed_dim=16,n_groups=4,num_inference_steps=1)
    elif model=='act3d':cfg['policy'].update(hidden_dim=32,nheads=4,enc_layers=1,dec_layers=1,dim_feedforward=64)
    else:cfg['policy'].update(n_layer=1,n_head=4,n_emb=32,num_inference_steps=1)
    return cfg


def episode(path, offset=0):
    with h5py.File(path,'w') as f:
        f.create_dataset('observations/qpos',data=np.arange(32,dtype=np.float32).reshape(8,4)+offset)
        ds=f.create_dataset('observations/depth/camera_top',data=np.full((8,8,8),500,np.uint16))
        ds.attrs.update(depth_scale=.001,aligned_to_rgb=True)
        f.create_dataset('observations/images/camera_top',data=np.full((8,8,8,3),128,np.uint8))


class GeometryTests(unittest.TestCase):
    def test_calibration_export_uses_supplied_frame_profile(self):
        from greenhouse_scara_3d_common.export_calibration import calibration_from_frame
        intr = SimpleNamespace(width=640, height=480, fx=421., fy=422., ppx=319., ppy=241.,
                               model='distortion.none', coeffs=[0.] * 5)
        profile = SimpleNamespace(get_intrinsics=lambda: intr)
        frame = SimpleNamespace(profile=SimpleNamespace(as_video_stream_profile=lambda: profile))
        result = calibration_from_frame(frame, 'test-camera', True)
        self.assertEqual(result['serial'], 'test-camera')
        self.assertEqual(result['intrinsics']['fx'], 421.)
        self.assertEqual(result['intrinsics']['cy'], 241.)
        self.assertTrue(result['aligned_to_rgb'])

    def test_live_adapter_matches_dataset_and_preserves_joint_history(self):
        from greenhouse_scara_3d_common.live import LivePointCloudSource
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episode_0.hdf5'; episode(path)
            cfg = config('dp3', tmp); cfg['pointcloud']['use_color'] = True
            ds = PointCloudDataset([path], cfg)
            packet = dict(depth_frames={'camera_top': np.full((8,8),500,np.uint16)},
                          depth_scales={'camera_top':.001},
                          frames={'camera_top':np.full((8,8,3),128,np.uint8)})
            with patch('greenhouse_scara_3d_common.live.MultiCameraSource') as cls:
                cls.return_value.read.return_value = packet
                live = LivePointCloudSource(cfg)
                first = live.read(np.arange(4))
                np.testing.assert_array_equal(first['agent_pos'][0], [np.arange(4), np.arange(4)])
                second = live.read(np.arange(4,8))
                np.testing.assert_array_equal(second['agent_pos'][0], ds[1]['obs']['agent_pos'])
                np.testing.assert_array_equal(second['point_cloud'][0], ds[1]['obs']['point_cloud'])
                live.close()
                cls.return_value.close.assert_called_once()

    def test_metric_geometry_mask_and_color(self):
        cam=calibration()['cameras']['camera_top']; opts=settings();opts['use_color']=True
        raw=np.full((8,8),500,np.uint16);raw[0,0]=0
        rgb=np.full((8,8,3),128,np.uint8)
        points=camera_points(raw,.001,cam,opts,rgb)
        self.assertEqual(points.shape,(63,6));self.assertEqual(points.dtype,np.float32)
        np.testing.assert_allclose(points[0,:3],[-.15625,-.21875,.5])
        np.testing.assert_allclose(points[:,3:],128/255,atol=1e-6)
        opts['max_depth_m']=.2
        with self.assertRaisesRegex(ValueError,'No valid points'):PointCloudBuilder(calibration(),opts)({'camera_top':raw},{'camera_top':.001},{'camera_top':rgb})

    def test_two_cameras_require_extrinsics_and_fuse(self):
        cal=calibration();cal['cameras']['camera_bottom']=copy.deepcopy(cal['cameras']['camera_top'])
        opt=settings();opt['cameras'].append('camera_bottom')
        with self.assertRaisesRegex(ValueError,'to_reference'):validate_calibration(cal,opt)
        transform=np.eye(4);transform[0,3]=1.
        cal['cameras']['camera_bottom']['to_reference']=transform.tolist()
        builder=PointCloudBuilder(cal,opt)
        depths={n:np.full((8,8),500,np.uint16) for n in opt['cameras']}
        pc=builder(depths,{n:.001 for n in depths})
        self.assertTrue(np.any(pc[:,0]>.7));self.assertTrue(np.any(pc[:,0]<.3))
        np.testing.assert_array_equal(pc,builder(depths,{n:.001 for n in depths}))

    def test_missing_intrinsics_and_alignment_rejected(self):
        cal=calibration();cal['cameras']['camera_top']['intrinsics']['fx']=None
        with self.assertRaisesRegex(ValueError,'intrinsics are missing'):PointCloudBuilder(cal,settings())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'episode_0.hdf5';episode(p)
            with h5py.File(p,'r+') as f:f['observations/depth/camera_top'].attrs['aligned_to_rgb']=False
            with self.assertRaisesRegex(ValueError,'alignment'):PointCloudDataset([p],config('dp3',tmp))

    def test_dataset_chunks_and_live_input_match_without_file_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'episode_0.hdf5';episode(path)
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            cfg=config('dp3',tmp);ds=PointCloudDataset([path],cfg)
            item=ds[3]
            with h5py.File(path,'r') as f:
                q=f['observations/qpos'][:]
                expected=PointCloudBuilder(cfg['calibration'],cfg['pointcloud'])(
                    {'camera_top':f['observations/depth/camera_top'][3]}, {'camera_top':.001})
            np.testing.assert_array_equal(item['obs']['point_cloud'][1],expected)
            np.testing.assert_array_equal(item['action'][:5],q[3:8])
            np.testing.assert_array_equal(item['action'][5:],np.repeat(q[-1:],3,axis=0))
            np.testing.assert_array_equal(item['obs']['agent_pos'],q[2:4])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_split_and_normalizer_use_training_episodes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'episode_0.hdf5';episode(p)
            cfg=config('dp3',tmp)
            with self.assertRaisesRegex(ValueError,'two episodes'):split_paths(cfg)
            other=Path(tmp)/'episode_4.hdf5';episode(other,1000)
            train_paths,val_paths=split_paths(cfg)
            self.assertFalse(set(train_paths)&set(val_paths))
            ds=PointCloudDataset([p],cfg);stats=ds.normalizer()['agent_pos'].get_input_stats()
            self.assertLess(float(stats['max'].max()),1000)


class ModelTests(unittest.TestCase):
    def test_available_models_losses_gradients_inference_and_rgb_points(self):
        for model in AVAILABLE_MODELS:
            for color in (False,True):
                with self.subTest(model=model,color=color), tempfile.TemporaryDirectory() as tmp:
                    cfg=config(model,tmp);cfg['pointcloud']['use_color']=color
                    p=Path(tmp)/'episode_0.hdf5';episode(p)
                    ds=PointCloudDataset([p],cfg)
                    batch=torch.utils.data.default_collate([ds[i] for i in range(4)])
                    policy=build_policy(cfg);policy.set_normalizer(ds.normalizer())
                    ema=copy.deepcopy(policy).eval().requires_grad_(False)
                    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3)
                    # DiTX starts with zero output/AdaLN gates: vision gradients appear after these open.
                    for _ in range(3 if model=='maniflow' else 1):
                        optimizer.zero_grad(set_to_none=True)
                        out=policy.compute_loss(batch,ema_model=ema) if model=='maniflow' else policy.compute_loss(batch)
                        loss=out[0] if isinstance(out,tuple) else out
                        self.assertTrue(torch.isfinite(loss));loss.backward()
                        optimizer.step()
                    encoder=policy.model.backbones[0] if model=='act3d' else policy.obs_encoder
                    grads=[p.grad for p in encoder.parameters() if p.requires_grad]
                    self.assertTrue(any(g is not None and torch.count_nonzero(g)>0 for g in grads))
                    policy.eval()
                    with torch.no_grad():pred=policy.predict_action(batch['obs'])['action']
                    self.assertEqual(tuple(pred.shape),(4,4,4));self.assertTrue(torch.isfinite(pred).all())

    def test_training_checkpoint_reload_and_resume_available_models(self):
        for model in AVAILABLE_MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                episode(Path(tmp)/'episode_0.hdf5');episode(Path(tmp)/'episode_1.hdf5',1)
                cfg=config(model,tmp)
                latest=train(cfg)
                policy,state=load_checkpoint(latest)
                self.assertEqual(state['global_step'],1)
                self.assertEqual(state['config']['calibration'],cfg['calibration'])
                cfg['resume']=True;cfg['epochs']=2
                train(cfg)
                _,state=load_checkpoint(latest)
                self.assertEqual(state['global_step'],2)
                self.assertEqual(state['epoch'],1)


if __name__=='__main__':unittest.main()
