"""Calibrated metric points from raw Z16 images; identical for training and live input."""
from functools import lru_cache
import numpy as np


def validate_calibration(calibration, settings):
    names = settings['cameras']
    if not names or len(set(names)) != len(names):
        raise ValueError('Choose one or more distinct pointcloud.cameras')
    if not isinstance(settings['num_points'], int) or settings['num_points'] < 1:
        raise ValueError('num_points must be a positive integer')
    near, far = settings.get('min_depth_m', 0), settings.get('max_depth_m')
    if not np.isfinite(near) or near < 0 or (far is not None and (not np.isfinite(far) or far <= near)):
        raise ValueError('Require finite min_depth_m >= 0; max_depth_m must be null or greater')
    candidates = settings.get('fps_candidates', 4096)
    stride = settings.get('pixel_stride', 1)
    if (not isinstance(stride, int) or stride < 1 or
            (candidates is not None and (not isinstance(candidates, int) or candidates < settings['num_points']))):
        raise ValueError('Integer pixel_stride >= 1; fps_candidates is null (all points) or >= num_points')
    reference = settings['reference_frame']
    for name in names:
        if name not in calibration.get('cameras', {}):
            raise ValueError(f'Missing calibration for {name}')
        camera = calibration['cameras'][name]
        intr = camera.get('intrinsics', {})
        values = [intr.get(k) for k in ('fx', 'fy', 'cx', 'cy')]
        if any(v is None for v in values):
            raise ValueError(f'{name}: camera intrinsics are missing. Run export_calibration.py with the recording camera/profile; do not guess them.')
        if not np.isfinite(values).all() or min(values[:2]) <= 0:
            raise ValueError(f'{name}: invalid camera intrinsics')
        if not isinstance(camera.get('aligned_to_rgb'), bool):
            raise ValueError(f'{name}: explicitly specify aligned_to_rgb')
        if min(camera.get('width', 0), camera.get('height', 0)) < 1:
            raise ValueError(f'{name}: calibration requires width/height')
        if intr.get('distortion_model') not in ('none', 'brown_conrady', 'inverse_brown_conrady'):
            raise ValueError(f'{name}: unsupported or missing distortion_model')
        if len(intr.get('coeffs', [])) != 5 or not np.isfinite(intr['coeffs']).all():
            raise ValueError(f'{name}: specify five finite distortion coefficients')
        transform = camera.get('to_reference')
        if transform is None:
            if name != reference:
                raise ValueError(f'{name}: to_reference is required to fuse points into {reference}; translation must be in metres')
        else:
            matrix = np.asarray(transform, dtype=float)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1])
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
                raise ValueError(f'{name}: to_reference must be a rigid 4x4 transform')
    low, high = settings.get('crop_min'), settings.get('crop_max')
    if (low is None) != (high is None):
        raise ValueError('Set both crop_min and crop_max, or neither')
    if low is not None and (np.shape(low) != (3,) or np.shape(high) != (3,)
                            or not np.isfinite([low, high]).all() or not np.all(np.asarray(low) < high)):
        raise ValueError('Crop bounds must be finite XYZ bounds in the reference frame')


@lru_cache(maxsize=16)
def _rays(width, height, fx, fy, cx, cy, model, coeffs, stride):
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    if model == 'none':
        rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    else:
        # Cache SDK deprojection so distortion is handled correctly once per profile.
        import pyrealsense2 as rs
        intr = rs.intrinsics()
        intr.width, intr.height = width, height
        intr.fx, intr.fy, intr.ppx, intr.ppy = fx, fy, cx, cy
        intr.model, intr.coeffs = getattr(rs.distortion, model), list(coeffs)
        rays = np.asarray([rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], 1.)
                           for x, y in zip(u.ravel(), v.ravel())]).reshape(*u.shape, 3)
    return np.asarray(rays, np.float32)


def camera_points(depth, scale, camera, settings, rgb=None):
    if depth.dtype != np.uint16 or depth.shape != (camera['height'], camera['width']):
        raise ValueError('Depth must be raw uint16 with the calibrated height and width')
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError('A positive depth_scale is required')
    stride = int(settings.get('pixel_stride', 1))
    z = depth[::stride, ::stride].astype(np.float32) * np.float32(scale)
    valid = np.isfinite(z) & (z > 0) & (z >= settings.get('min_depth_m', 0))
    if settings.get('max_depth_m') is not None:
        valid &= z <= settings['max_depth_m']
    intr = camera['intrinsics']
    rays = _rays(camera['width'], camera['height'], intr['fx'], intr['fy'], intr['cx'], intr['cy'],
                 intr['distortion_model'], tuple(intr['coeffs']), stride)
    xyz = (rays * z[..., None])[valid]
    if camera.get('to_reference') is not None:
        transform = np.asarray(camera['to_reference'], np.float32)
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    if settings.get('use_color', False):
        if not camera['aligned_to_rgb']:
            raise ValueError('XYZRGB requires aligned depth; raw depth/RGB registration is not implicit')
        if rgb is None or rgb.dtype != np.uint8 or rgb.shape != (*depth.shape, 3):
            raise ValueError('XYZRGB requires matching uint8 RGB images')
        xyz = np.concatenate([xyz, rgb[::stride, ::stride][valid].astype(np.float32) / 255.], -1)
    return xyz.astype(np.float32)


def sample_points(points, count, candidates=4096):
    """Deterministic FPS in metres; candidates=None uses the complete point cloud."""
    if not len(points):
        raise ValueError('No valid points after depth filtering/cropping; check calibration and bounds')
    if len(points) <= count:
        return points[np.arange(count) % len(points)].copy()
    if candidates is not None and len(points) > candidates:
        points = points[np.linspace(0, len(points) - 1, candidates, dtype=np.int64)]
    xyz = points[:, :3]
    distances = np.full(len(points), np.inf, dtype=np.float32)
    index = int(np.argmax(np.sum((xyz - xyz.mean(0)) ** 2, axis=1)))
    chosen = np.empty(count, dtype=np.int64)
    for i in range(count):
        chosen[i] = index
        distances = np.minimum(distances, np.sum((xyz - xyz[index]) ** 2, axis=1))
        distances[chosen[:i + 1]] = -1
        index = int(np.argmax(distances))
    return np.ascontiguousarray(points[chosen], dtype=np.float32)


class PointCloudBuilder:
    def __init__(self, calibration, settings):
        validate_calibration(calibration, settings)
        self.calibration, self.settings = calibration, settings

    def __call__(self, depths, scales, rgbs=None):
        points = self.dense(depths, scales, rgbs)
        return sample_points(points, self.settings['num_points'], self.settings.get('fps_candidates', 4096))

    def dense(self, depths, scales, rgbs=None):
        """Cloud before FPS; shared by training and the inspection notebook."""
        settings = self.settings
        points = np.concatenate([camera_points(depths[name], scales[name], self.calibration['cameras'][name],
                                 settings, (rgbs or {}).get(name)) for name in settings['cameras']])
        if settings.get('crop_min') is not None:
            xyz = points[:, :3]
            points = points[((xyz >= settings['crop_min']) & (xyz <= settings['crop_max'])).all(-1)]
        return points

    def from_hdf5(self, root, index):
        points = self.dense_from_hdf5(root, index)
        return sample_points(points, self.settings['num_points'], self.settings.get('fps_candidates', 4096))

    def dense_from_hdf5(self, root, index):
        depths, scales, rgbs = {}, {}, {}
        for name in self.settings['cameras']:
            ds = root[f'observations/depth/{name}']
            camera = self.calibration['cameras'][name]
            if 'aligned_to_rgb' not in ds.attrs or bool(ds.attrs['aligned_to_rgb']) != camera['aligned_to_rgb']:
                raise ValueError(f'{name}: HDF5 alignment does not match calibration')
            depths[name], scales[name] = ds[index], ds.attrs.get('depth_scale')
            if self.settings.get('use_color', False): rgbs[name] = root[f'observations/images/{name}'][index]
        return self.dense(depths, scales, rgbs)
