import warnings
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
from greenhouse_scara_common.data import (episode_paths, image_specs, validate_episode,
    action_dataset, read_images)


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, paths, shape_meta, norm_stats, chunk_size, action_source='auto', action_offset=1):
        self.paths = paths
        self.specs = image_specs(shape_meta)
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.action_source = action_source
        self.action_offset = action_offset

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with h5py.File(self.paths[index], 'r') as root:
            actions = action_dataset(root, self.action_source, self.action_offset)
            start = np.random.randint(actions.shape[0])
            qpos = np.asarray(root['observations/qpos'][start], dtype=np.float32)
            images = {key: torch.from_numpy(read_images(root, key, spec, start, 1)[0])
                      for key, spec in self.specs.items()}
            action = np.asarray(actions[start:start + self.chunk_size], dtype=np.float32)
        count = len(action)
        padded = np.zeros((self.chunk_size, 4), dtype=np.float32)
        padded[:count] = action
        is_pad = np.arange(self.chunk_size) >= count
        padded = (padded - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        qpos = (qpos - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']
        return images, torch.from_numpy(qpos), torch.from_numpy(padded), torch.from_numpy(is_pad)


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
              chunk_size, inputs_config, num_workers=4):
    paths = episode_paths(dataset_dir)
    cfg = inputs_config
    for path in paths:
        with h5py.File(path, 'r') as root:
            validate_episode(root, cfg['shape_meta'], cfg.get('action_source', 'auto'), cfg.get('action_offset', 1))
    indices = np.random.permutation(len(paths))
    if len(paths) == 1:
        if not cfg.get('allow_single_episode', False):
            raise ValueError('Need two demos for held-out validation; allow_single_episode=true is only for smoke tests')
        warnings.warn('Single-demo smoke test: training and validation use the SAME episode', stacklevel=2)
        train = validation = paths
    else:
        cut = min(len(paths) - 1, max(1, int(.8 * len(paths))))
        train, validation = ([paths[i] for i in indices[:cut]], [paths[i] for i in indices[cut:]])
    qpos_values, action_values = [], []
    source, offset = cfg.get('action_source', 'auto'), cfg.get('action_offset', 1)
    for path in train:
        with h5py.File(path, 'r') as root:
            qpos_values.append(root['observations/qpos'][:])
            action_values.append(action_dataset(root, source, offset)[:])
    stats = {}
    for name, values in [('qpos', qpos_values), ('action', action_values)]:
        data = np.concatenate(values).astype(np.float32)
        stats[name + '_mean'] = data.mean(0)
        stats[name + '_std'] = np.maximum(data.std(0), .01)
    stats['example_qpos'] = qpos_values[0]
    options = {'num_workers': num_workers, 'pin_memory': True}
    if num_workers:
        options.update(prefetch_factor=2, persistent_workers=True)
    datasets = [EpisodicDataset(items, cfg['shape_meta'], stats, chunk_size, source, offset)
                for items in (train, validation)]
    return (DataLoader(datasets[0], batch_size=batch_size_train, shuffle=True, **options),
            DataLoader(datasets[1], batch_size=batch_size_val, shuffle=False, **options), stats)


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
