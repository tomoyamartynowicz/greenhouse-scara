"""Read-only episode access and identical RGB/depth preprocessing for all policies."""
from pathlib import Path
import re
import cv2
import numpy as np


def episode_paths(folder):
    paths = [p for p in Path(folder).expanduser().glob('episode_*.hdf5')
             if re.fullmatch(r'episode_\d+\.hdf5', p.name)]
    paths.sort(key=lambda p: int(p.stem.split('_')[-1]))
    if not paths:
        raise FileNotFoundError(f'No episode_*.hdf5 files in {folder}')
    return paths


def image_specs(shape_meta):
    specs = {str(k): dict(v) for k, v in shape_meta['obs'].items() if v.get('type') in ('rgb', 'depth')}
    if not specs:
        raise ValueError('Select at least one RGB or depth stream')
    for key, spec in specs.items():
        channels = 3 if spec['type'] == 'rgb' else 2
        if len(spec['shape']) != 3 or int(spec['shape'][0]) != channels or min(spec['shape'][1:]) < 1:
            raise ValueError(f'{key}: expected shape [{channels}, H, W]')
        if spec['type'] == 'depth' and (not np.isfinite(spec.get('max_depth_m', .5)) or spec.get('max_depth_m', .5) <= 0):
            raise ValueError(f'{key}: max_depth_m must be positive')
        if spec['type'] == 'depth' and 'depth_normalization_m' in spec:
            scale = spec['depth_normalization_m']
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'{key}: depth_normalization_m must be positive')
            if 'max_depth_m' in spec:
                raise ValueError(f'{key}: choose depth_normalization_m or legacy max_depth_m')
    return specs


def image_path(key, spec):
    group = 'images' if spec['type'] == 'rgb' else 'depth'
    return f"observations/{group}/{spec.get('camera', key)}"


def action_dataset(root, source='auto', offset=1):
    """Array-like view; shift before slicing/padding so chunks retain the next action."""
    if source not in ('auto', 'stored', 'next_qpos'):
        raise ValueError('action_source must be auto, stored or next_qpos')
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
        raise ValueError('action_offset must be a positive integer')
    qpos = root.get('observations/qpos')
    if qpos is None or qpos.ndim != 2 or qpos.shape[1] != 4 or not len(qpos):
        raise ValueError(f'{root.filename}: nonempty observations/qpos (T,4) is required; camera-only data cannot train a policy')
    if source == 'stored' or source == 'auto' and 'action' in root:
        if 'action' not in root:
            raise ValueError(f'{root.filename}: stored action is missing')
        actions = root['action']
        if actions.shape != qpos.shape:
            raise ValueError(f'{root.filename}: action and qpos shapes must match')
        return actions
    return ShiftedQpos(qpos, offset)


class ShiftedQpos:
    def __init__(self, qpos, offset):
        self.qpos = qpos
        self.offset = offset
        self.shape = qpos.shape
        self.ndim = qpos.ndim

    def __getitem__(self, index):
        if not isinstance(index, slice):
            i = int(index)
            if i < 0: i += self.shape[0]
            if not 0 <= i < self.shape[0]: raise IndexError(index)
            return self.qpos[min(i + self.offset, self.shape[0] - 1)]
        start, stop, step = index.indices(self.shape[0])
        if step != 1: raise ValueError('Only contiguous action slices are supported')
        count = max(0, stop - start)
        if not count: return np.empty((0, 4), dtype=np.float32)
        start = min(start + self.offset, self.shape[0] - 1)
        stop = min(start + count, self.shape[0])
        values = np.asarray(self.qpos[start:stop], dtype=np.float32)
        if len(values) < count:
            values = np.concatenate([values, np.repeat(values[-1:], count - len(values), axis=0)])
        return values


def read_padded(dataset, start, length):
    a, b = max(start, 0), min(start + length, dataset.shape[0])
    if a >= b: raise IndexError('Window does not overlap episode')
    values = np.asarray(dataset[a:b], dtype=np.float32)
    return np.concatenate([np.repeat(values[:1], a - start, axis=0), values,
                           np.repeat(values[-1:], start + length - b, axis=0)])


def preprocess_image(frame, spec, depth_scale=None):
    """RGB: [0,1]. Depth: [scaled metres, valid mask].

    depth_normalization_m preserves all distances. Configurations without it
    retain legacy max_depth_m clipping for existing policies/checkpoints.
    """
    height, width = map(int, spec['shape'][1:])
    if spec['type'] == 'rgb':
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError('RGB must be uint8 HWC with 3 channels')
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(np.moveaxis(frame, -1, 0), dtype=np.float32) / 255
    if frame.ndim != 2 or frame.dtype != np.uint16:
        raise ValueError('Depth must be a uint16 HW image')
    if depth_scale is None or not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError('Depth requires a positive depth_scale attribute (metres per raw unit)')
    # Nearest-neighbour preserves invalid zeros and discontinuities.
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    valid = (frame > 0).astype(np.float32)
    depth = frame.astype(np.float32) * np.float32(depth_scale)
    if 'depth_normalization_m' in spec:
        depth = depth / np.float32(spec['depth_normalization_m'])
    else:
        depth = np.clip(depth / np.float32(spec.get('max_depth_m', .5)), 0, 1)
    return np.ascontiguousarray(np.stack([depth, valid]), dtype=np.float32)


def read_images(root, key, spec, start, length):
    dataset = root[image_path(key, spec)]
    a, b = max(start, 0), min(start + length, len(dataset))
    if a >= b: raise IndexError('Image window does not overlap episode')
    values = np.stack([preprocess_image(frame, spec, dataset.attrs.get('depth_scale'))
                       for frame in dataset[a:b]])
    return np.concatenate([np.repeat(values[:1], a - start, axis=0), values,
                           np.repeat(values[-1:], start + length - b, axis=0)])


def validate_episode(root, shape_meta, action_source='auto', action_offset=1):
    actions = action_dataset(root, action_source, action_offset)
    length = actions.shape[0]
    if not np.isfinite(root['observations/qpos'][:]).all() or not np.isfinite(actions[:]).all():
        raise ValueError(f'{root.filename}: joints/actions contain NaN or Inf')
    for key, spec in image_specs(shape_meta).items():
        path = image_path(key, spec)
        if path not in root: raise ValueError(f'{root.filename}: missing requested stream {path}')
        image = root[path]
        dims = 4 if spec['type'] == 'rgb' else 3
        dtype = np.uint8 if spec['type'] == 'rgb' else np.uint16
        if image.ndim != dims or image.shape[0] != length or image.dtype != dtype or min(image.shape[1:]) < 1:
            raise ValueError(f'{root.filename}: invalid {path} shape/dtype or frame count')
        if spec['type'] == 'rgb' and image.shape[-1] != 3:
            raise ValueError(f'{root.filename}: RGB requires 3 channels')
        if spec['type'] == 'depth':
            scale = image.attrs.get('depth_scale')
            if scale is None or not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'{root.filename}: {path} needs a positive depth_scale')
            if 'aligned_to_rgb' in spec and bool(image.attrs.get('aligned_to_rgb', False)) != bool(spec['aligned_to_rgb']):
                raise ValueError(f'{root.filename}: {path} alignment differs from the input configuration')
    return length
