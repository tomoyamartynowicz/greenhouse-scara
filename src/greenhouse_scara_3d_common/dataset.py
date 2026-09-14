"""Shared read-only HDF5 windows and episode-level splits for all four 3D models."""
from collections import OrderedDict
import warnings
import h5py
import numpy as np
import torch
from greenhouse_scara_common.data import action_dataset, episode_paths, read_padded
from .pointcloud import PointCloudBuilder
from .vendor.dp3.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, paths, cfg):
        self.paths = list(paths)
        self.cfg = cfg
        self.builder = PointCloudBuilder(cfg['calibration'], cfg['pointcloud'])
        self.horizon, self.n_obs = cfg['horizon'], cfg['n_obs_steps']
        self.samples, self.lengths = [], []
        self.cache = OrderedDict()
        for episode, path in enumerate(paths):
            with h5py.File(path, 'r') as root:
                actions = action_dataset(root, cfg['action_source'], cfg['action_offset'])
                length = actions.shape[0]
                if not np.isfinite(actions[:]).all() or not np.isfinite(root['observations/qpos'][:]).all():
                    raise ValueError(f'{path}: nonfinite joints/actions')
                for name in cfg['pointcloud']['cameras']:
                    key = f'observations/depth/{name}'
                    if key not in root: raise ValueError(f'{path}: missing {key}')
                    ds = root[key]
                    cam = cfg['calibration']['cameras'][name]
                    if ds.shape != (length, cam['height'], cam['width']) or ds.dtype != np.uint16:
                        raise ValueError(f'{path}: depth length, size or dtype does not match joints/calibration')
                    if cfg['pointcloud'].get('use_color', False):
                        rgb = root.get(f'observations/images/{name}')
                        if rgb is None or rgb.shape != (*ds.shape, 3) or rgb.dtype != np.uint8:
                            raise ValueError(f'{path}: invalid or missing RGB for {name}')
                # Validate geometry without expensive FPS for every episode at startup.
                if not len(self.builder.dense_from_hdf5(root, 0)):
                    raise ValueError(f'{path}: no valid points after depth filtering/cropping')
                self.lengths.append(length)
                self.samples.extend((episode, anchor) for anchor in range(length))

    def __len__(self): return len(self.samples)

    def _points(self, episode, frame, root):
        key = (episode, frame)
        if key not in self.cache:
            self.cache[key] = self.builder.from_hdf5(root, frame)
            if len(self.cache) > 128: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, index):
        episode, anchor = self.samples[index]
        start = anchor - self.n_obs + 1
        length = self.lengths[episode]
        with h5py.File(self.paths[episode], 'r') as root:
            points = np.stack([self._points(episode, max(0, start + i), root) for i in range(self.n_obs)])
            qpos = read_padded(root['observations/qpos'], start, self.n_obs)
            actions = read_padded(action_dataset(root, self.cfg['action_source'], self.cfg['action_offset']), start, self.horizon)
        padded = (np.arange(start, start + self.horizon) < 0) | (np.arange(start, start + self.horizon) >= length)
        return {'obs': {'point_cloud': torch.from_numpy(points), 'agent_pos': torch.from_numpy(qpos)},
                'action': torch.from_numpy(actions), 'is_pad': torch.from_numpy(padded)}

    def normalizer(self):
        # Fit only training episodes. Legacy models can retain metric XYZ unchanged.
        values = {'agent_pos': [], 'action': []}
        for path in self.paths:
            with h5py.File(path, 'r') as root:
                values['agent_pos'].append(root['observations/qpos'][:])
                values['action'].append(action_dataset(root, self.cfg['action_source'], self.cfg['action_offset'])[:])
        norm = LinearNormalizer()
        mode = 'gaussian' if self.cfg['model'] == 'act3d' else 'limits'
        for key, arrays in values.items():
            norm[key] = SingleFieldLinearNormalizer.create_fit(np.concatenate(arrays).astype(np.float32), mode=mode, last_n_dims=1)
        mode = self.cfg['pointcloud'].get('normalization', 'identity')
        if mode == 'identity':
            norm['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        elif mode == 'limits':
            # Same per-channel [-1, 1] scaling as DP3, streaming bounds to avoid
            # materializing all clouds or running FPS over every frame here.
            low = high = None
            for index, path in enumerate(self.paths):
                print(f'Pointcloud normalization {index + 1}/{len(self.paths)}: {path.name}', flush=True)
                with h5py.File(path, 'r') as root:
                    for frame in range(len(root['observations/qpos'])):
                        points = self.builder.dense_from_hdf5(root, frame)
                        if not len(points):
                            raise ValueError(f'{path}, frame {frame}: no valid points')
                        lo, hi = points.min(0), points.max(0)
                        low = lo if low is None else np.minimum(low, lo)
                        high = hi if high is None else np.maximum(high, hi)
            norm['point_cloud'] = SingleFieldLinearNormalizer.create_fit(
                np.stack([low, high]), mode='limits', last_n_dims=1)
        else:
            raise ValueError('pointcloud.normalization must be identity or limits')
        return norm


def split_paths(cfg):
    paths = episode_paths(cfg['dataset_dir'])
    if len(paths) == 1:
        if not cfg.get('allow_single_episode', False):
            raise ValueError('Need at least two episodes; allow_single_episode=true is only for technical tests')
        warnings.warn('Single episode: training and validation use the SAME demo, not held-out validation', stacklevel=2)
        return paths, paths
    indices = np.random.default_rng(cfg['seed']).permutation(len(paths))
    n_val = min(len(paths) - 1, max(1, round(len(paths) * cfg['val_ratio'])))
    return [paths[i] for i in indices[n_val:]], [paths[i] for i in indices[:n_val]]
