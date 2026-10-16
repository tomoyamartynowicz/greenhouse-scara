"""Reusable, atomic pointcloud sidecars; original demonstrations stay read-only."""
import hashlib
import json
from pathlib import Path
import tempfile
import h5py
import numpy as np
from .pointcloud import PointCloudBuilder


def geometry_signature(cfg):
    payload = dict(version=1, calibration=cfg['calibration'], pointcloud=cfg['pointcloud'])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_path(source, cfg):
    directory = cfg.get('pointcloud_cache_dir')
    return Path(directory) / Path(source).name if directory else None


def validate_cache(source, cfg):
    path = cache_path(source, cfg)
    if path is None or not path.is_file():
        raise FileNotFoundError(f'Pointcloud cache missing: {path}; run scripts/prepare_3d.py first')
    info = Path(source).stat()
    with h5py.File(path, 'r') as f:
        if (f.attrs.get('geometry_signature') != geometry_signature(cfg)
                or f.attrs.get('source_size') != info.st_size
                or f.attrs.get('source_mtime_ns') != info.st_mtime_ns
                or f.attrs.get('source_name') != Path(source).name):
            raise ValueError(f'{path}: cache differs from source/calibration/settings; prepare a new cache directory')
        shape = (cfg['pointcloud']['num_points'], 6 if cfg['pointcloud']['use_color'] else 3)
        if f['point_cloud'].shape[1:] != shape or not f.attrs.get('complete', False):
            raise ValueError(f'{path}: incomplete or incompatible pointcloud cache')
    return path


def prepare_episode(source, cfg):
    source = Path(source)
    destination = cache_path(source, cfg)
    if destination is None:
        raise ValueError('Set pointcloud_cache_dir')
    if destination.resolve() == source.resolve():
        raise ValueError('Cache directory must differ from the original dataset')
    if destination.exists():
        validate_cache(source, cfg)
        return str(destination), 'existing'
    builder = PointCloudBuilder(cfg['calibration'], cfg['pointcloud'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = source.stat()
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.pointcloud-') as tmp:
        staged = Path(tmp) / source.name
        with h5py.File(source, 'r') as data, h5py.File(staged, 'w') as cache:
            count = len(data['observations/qpos'])
            channels = 6 if cfg['pointcloud']['use_color'] else 3
            points = cache.create_dataset('point_cloud', (count, cfg['pointcloud']['num_points'], channels),
                                          dtype='float32', chunks=(1, cfg['pointcloud']['num_points'], channels),
                                          compression='lzf')
            low = np.full(channels, np.inf); high = -low.copy()
            for frame in range(count):
                try:
                    cloud = builder.from_hdf5(data, frame)
                except ValueError as exc:
                    raise ValueError(f'{source.name}, frame {frame}: {exc}') from exc
                points[frame] = cloud
                low = np.minimum(low, cloud.min(0)); high = np.maximum(high, cloud.max(0))
            if (source.stat().st_size, source.stat().st_mtime_ns) != (info.st_size, info.st_mtime_ns):
                raise RuntimeError(f'{source} changed while building cache')
            cache.attrs.update(geometry_signature=geometry_signature(cfg), source_name=source.name,
                               source_size=info.st_size, source_mtime_ns=info.st_mtime_ns,
                               minimum=low, maximum=high, complete=True)
            cache.attrs['geometry_json'] = json.dumps(dict(calibration=cfg['calibration'], pointcloud=cfg['pointcloud']))
        staged.replace(destination)
    return str(destination), 'created'
