import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import h5py
import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2]
for folder in (SRC, SRC/'greenhouse_scara_act', SRC/'greenhouse_scara_diffusion_policy', SRC/'greenhouse_scara_flow_matching_policy'):
    sys.path.insert(0, str(folder))
from greenhouse_scara_common.data import action_dataset, preprocess_image, validate_episode
from greenhouse_scara_common.live import observation_from_history, MultiCameraSource
from greenhouse_scara_common.vision import MultiModalObsEncoder
from greenhouse_diffusion_policy.dataset.scara_image_dataset import ScaraImageDataset as DPData
from greenhouse_flow_matching_policy.dataset.scara_image_dataset import ScaraImageDataset as FMData
spec = importlib.util.spec_from_file_location('greenhouse_act_data', SRC/'greenhouse_scara_act/utils.py')
act_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(act_data)


def shape_meta():
    obs = {}
    for camera in ('camera_top', 'camera_bottom'):
        obs[camera+'_rgb'] = {'camera':camera,'type':'rgb','shape':[3,32,32]}
        obs[camera+'_depth'] = {'camera':camera,'type':'depth','shape':[2,32,32],
                                'max_depth_m':.5,'aligned_to_rgb':True}
    obs['qpos'] = {'type':'low_dim','shape':[4]}
    return {'obs':obs,'action':{'shape':[4]}}


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'episode_7.hdf5'
        self.meta = shape_meta()
        with h5py.File(self.path,'w') as f:
            f.create_dataset('observations/qpos',data=np.arange(24,dtype=np.float32).reshape(6,4))
            for camera in ('camera_top','camera_bottom'):
                f.create_dataset('observations/images/'+camera,data=np.full((6,32,32,3),128,np.uint8))
                raw = np.full((6,32,32),1000,np.uint16)
                raw[:,:,0]=0
                d=f.create_dataset('observations/depth/'+camera,data=raw)
                d.attrs['depth_scale']=np.float64(.0001)
                d.attrs['aligned_to_rgb']=True

    def test_actions_shift_before_chunking_and_repeat_last(self):
        with h5py.File(self.path,'r') as f:
            q=f['observations/qpos'][:]
            actions=action_dataset(f)
            np.testing.assert_array_equal(actions[:3],q[1:4])
            np.testing.assert_array_equal(actions[4:6],q[[5,5]])
            np.testing.assert_array_equal(action_dataset(f,offset=2)[3:6],q[[5,5,5]])
            with self.assertRaises(ValueError): action_dataset(f,source='stored')
        with h5py.File(self.path,'r+') as f:
            f.create_dataset('action',data=np.full((6,4),99,np.float32))
        with h5py.File(self.path,'r') as f:
            np.testing.assert_array_equal(action_dataset(f)[:],99)
            np.testing.assert_array_equal(action_dataset(f,'next_qpos')[:2],q[1:3])

    def test_all_three_loaders_read_multicamera_depth_without_mutating_hdf5(self):
        before=hashlib.sha256(self.path.read_bytes()).hexdigest()
        for cls in (DPData,FMData):
            ds=cls(self.meta,self.temp.name,horizon=4,n_obs_steps=2,pad_before=1,pad_after=2,
                   val_ratio=0,allow_single_episode=True)
            ds.samples=[(0,1),(0,4)]
            first=ds[0]
            self.assertEqual(set(first['obs']),set(self.meta['obs']))
            for camera in ('camera_top','camera_bottom'):
                depth=first['obs'][camera+'_depth']
                self.assertEqual(depth.dtype,torch.float32)
                self.assertEqual(tuple(depth.shape),(2,2,32,32))
                self.assertTrue(torch.allclose(depth[:,0,:,1:],torch.full((2,32,31),.2)))
                self.assertTrue(torch.all(depth[:,:, :,0]==0))
            with h5py.File(self.path,'r') as f:
                q=f['observations/qpos'][:]
            np.testing.assert_array_equal(first['action'],q[2:6])
            np.testing.assert_array_equal(ds[1]['action'],q[[5,5,5,5]])
            normalized=ds.get_normalizer().normalize(first['obs'])
            torch.testing.assert_close(normalized['camera_top_depth'],first['obs']['camera_top_depth'])
        stats={key:np.zeros(4,np.float32) if key.endswith('mean') else np.ones(4,np.float32)
               for key in ('qpos_mean','qpos_std','action_mean','action_std')}
        ds=act_data.EpisodicDataset([self.path],self.meta,stats,4)
        with patch.object(np.random,'randint',return_value=4):
            images,qpos,actions,pad=ds[0]
        np.testing.assert_array_equal(actions[:2],q[[5,5]])
        np.testing.assert_array_equal(pad,[False,False,True,True])
        self.assertEqual(set(images),set(self.meta['obs'])-{'qpos'})
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),before)

    def test_live_and_dataset_preprocessing_match(self):
        packet={'frames':{},'depth_frames':{},'depth_scales':{}}
        with h5py.File(self.path,'r') as f:
            q=f['observations/qpos'][0]
            for name in ('camera_top','camera_bottom'):
                packet['frames'][name]=f['observations/images/'+name][0]
                packet['depth_frames'][name]=f['observations/depth/'+name][0]
                packet['depth_scales'][name]=f['observations/depth/'+name].attrs['depth_scale']
        specs={k:v for k,v in self.meta['obs'].items() if k!='qpos'}
        live=observation_from_history([(packet,q)],specs,'cpu')
        data=DPData(self.meta,self.temp.name,horizon=4,n_obs_steps=1,val_ratio=0)
        data.samples=[(0,0)]
        for key,value in data[0]['obs'].items():
            torch.testing.assert_close(live[key][0],value)

    def test_act_unclipped_depth_matches_live(self):
        for spec in self.meta['obs'].values():
            if spec['type'] == 'depth':
                del spec['max_depth_m']
                spec['depth_normalization_m'] = 1.0
        packet = {'frames': {}, 'depth_frames': {}, 'depth_scales': {}}
        with h5py.File(self.path, 'r+') as f:
            qpos = f['observations/qpos'][0]
            for name in ('camera_top', 'camera_bottom'):
                raw = np.full((32, 32), 30000, np.uint16)
                raw[:, 0] = 0
                raw[:, 1] = 20000
                f['observations/depth/' + name][0] = raw
                packet['frames'][name] = f['observations/images/' + name][0]
                packet['depth_frames'][name] = raw
                packet['depth_scales'][name] = .0001
        stats = {key: np.zeros(4, np.float32) if key.endswith('mean') else np.ones(4, np.float32)
                 for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std')}
        dataset = act_data.EpisodicDataset([self.path], self.meta, stats, 4)
        with patch.object(np.random, 'randint', return_value=0):
            images, _, _, _ = dataset[0]
        live = observation_from_history([(packet, qpos)], dataset.specs, 'cpu')
        for key, value in images.items():
            torch.testing.assert_close(live[key][0, 0], value)
            if dataset.specs[key]['type'] == 'depth':
                torch.testing.assert_close(value[0, 0], torch.tensor([0., 2.] + [3.] * 30))
                torch.testing.assert_close(value[1, 0], torch.tensor([0.] + [1.] * 31))

    def test_act_targets_have_no_delay_compensation(self):
        stats = {key: np.zeros(4, np.float32) if key.endswith('mean') else np.ones(4, np.float32)
                 for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std')}
        dataset = act_data.EpisodicDataset([self.path], self.meta, stats, 3)
        with h5py.File(self.path, 'r') as f:
            qpos = f['observations/qpos'][:]
        with patch.object(np.random, 'randint', return_value=2):
            _, observation, actions, pad = dataset[0]
        np.testing.assert_array_equal(observation, qpos[2])
        np.testing.assert_array_equal(actions, qpos[3:6])
        self.assertFalse(pad.any())
        with h5py.File(self.path, 'r+') as f:
            f.create_dataset('action', data=qpos + 100)
        with patch.object(np.random, 'randint', return_value=2):
            _, _, actions, _ = dataset[0]
        np.testing.assert_array_equal(actions, qpos[2:5] + 100)

    def test_rgb_only_and_mixed_camera_selection(self):
        for keys in (('camera_top_rgb',),
                     ('camera_top_rgb', 'camera_bottom_rgb', 'camera_bottom_depth')):
            meta = {'obs': {key: self.meta['obs'][key] for key in (*keys, 'qpos')},
                    'action': self.meta['action']}
            for cls in (DPData, FMData):
                ds = cls(meta, self.temp.name, horizon=4, n_obs_steps=1, val_ratio=0)
                self.assertEqual(set(ds[0]['obs']), set(keys) | {'qpos'})
            stats = {key: np.zeros(4, np.float32) if key.endswith('mean') else np.ones(4, np.float32)
                     for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std')}
            images, _, _, _ = act_data.EpisodicDataset([self.path], meta, stats, 4)[0]
            self.assertEqual(set(images), set(keys))

    def test_validation_requires_requested_depth_scale_joints_and_alignment(self):
        with h5py.File(self.path,'r+') as f:
            del f['observations/depth/camera_top'].attrs['depth_scale']
            with self.assertRaisesRegex(ValueError,'depth_scale'): validate_episode(f,self.meta)
            f['observations/depth/camera_top'].attrs['depth_scale']=.0001
            f['observations/depth/camera_top'].attrs['aligned_to_rgb']=False
            with self.assertRaisesRegex(ValueError,'alignment'): validate_episode(f,self.meta)
            del f['observations/qpos']
            with self.assertRaisesRegex(ValueError,'camera-only'):validate_episode(f,self.meta)

    def test_two_camera_live_adapter_reads_both_and_closes(self):
        specs={k:v for k,v in self.meta['obs'].items() if k!='qpos'}
        cameras=[]
        for name in ('camera_top','camera_bottom'):
            camera=Mock()
            camera.read.return_value={'frames':{name:np.zeros((32,32,3),np.uint8)},
                 'depth_frames':{name:np.ones((32,32),np.uint16)},'depth_scale':.0001}
            cameras.append(camera)
        with patch('greenhouse_scara_common.realsense.RealSenseSource',side_effect=cameras):
            source=MultiCameraSource(specs).start()
            packet=source.read()
            source.close()
        self.assertEqual(set(packet['frames']),{'camera_top','camera_bottom'})
        self.assertEqual(set(packet['depth_frames']),{'camera_top','camera_bottom'})
        for camera in cameras: camera.close.assert_called_once()

    def test_multimodal_encoder_backpropagates_to_each_stream(self):
        encoder=MultiModalObsEncoder(self.meta)
        obs={key:torch.rand(2,*spec['shape']) for key,spec in self.meta['obs'].items()}
        output=encoder(obs)
        self.assertEqual(tuple(output.shape),(2,2052))
        output.square().mean().backward()
        for model in encoder.models.values():
            self.assertIsNotNone(model.conv1.weight.grad)
            self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())


if __name__=='__main__':unittest.main()
