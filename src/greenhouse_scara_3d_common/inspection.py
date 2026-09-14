"""Inspect the exact calibrated point pipeline used for DP3 training."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from .pointcloud import PointCloudBuilder, sample_points


def inspect_frame(episode, calibration, settings, frame=0):
    if len(settings['cameras']) != 1:
        raise ValueError('This viewer currently displays one camera at a time')
    camera = settings['cameras'][0]
    colored = copy.deepcopy(settings)
    colored['use_color'] = True
    builder = PointCloudBuilder(calibration, colored)
    with h5py.File(episode, 'r') as root:
        ds = root[f'observations/depth/{camera}']
        if not 0 <= frame < len(ds):
            raise IndexError(f'Frame must be 0..{len(ds)-1}')
        raw = ds[frame]
        scale = float(ds.attrs['depth_scale'])
        rgb = root[f'observations/images/{camera}'][frame]
        full = builder.dense_from_hdf5(root, frame)
        qpos = root['observations/qpos'][frame]
    sampled = sample_points(full, settings['num_points'], settings.get('fps_candidates', 4096))
    depth = raw.astype(np.float32) * scale
    valid = depth[raw > 0]
    report = dict(episode=str(Path(episode).resolve()), frame=int(frame), camera=camera,
                  reference_frame=settings['reference_frame'], depth_scale=scale,
                  pixels=int(raw.size), zero_depth_pixels=int((raw == 0).sum()),
                  max_code_pixels=int((raw == 65535).sum()),
                  valid_depth_percentiles_m=dict(zip(['min', 'p01', 'median', 'p95', 'p99', 'max'],
                      np.percentile(valid, [0, 1, 50, 95, 99, 100]).tolist())),
                  dense_points=len(full), model_points=len(sampled),
                  xyz_min_m=full[:, :3].min(0).tolist(), xyz_max_m=full[:, :3].max(0).tolist(),
                  qpos=qpos.tolist(), settings=settings,
                  notes=['No depth-to-color transform is applied twice to already aligned depth.',
                         'Zero depth is invalid; max-code values are reported, not silently discarded.',
                         'Model points shown in metres, before the stored training normalizer.'])
    return dict(rgb=rgb, depth=depth, full=full, sampled=sampled, report=report)


def pointcloud_figure(result, max_display=20000):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    full, sampled = result['full'], result['sampled']
    shown = full[np.linspace(0, len(full) - 1, min(max_display, len(full)), dtype=int)]
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=[f'Volledige observatie: {len(full):,} punten '
                                         f'({len(shown):,} getoond)', f'DP3-input: {len(sampled):,} punten'])
    for col, points in enumerate((shown, sampled), 1):
        colors = (points[:, 3:6] * 255).round().clip(0, 255).astype(np.uint8)
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                      marker=dict(size=1 if col == 1 else 2, color=[f'rgb({r},{g},{b})' for r,g,b in colors]),
                      hovertemplate='X %{x:.3f} m<br>Y %{y:.3f} m<br>Z %{z:.3f} m<extra></extra>',
                      showlegend=False), row=1, col=col)
    low, high = full[:, :3].min(0), full[:, :3].max(0)
    axes = {axis+'axis': dict(title=title, range=[float(lo), float(hi)])
            for axis, title, lo, hi in zip('xyz', ['X rechts (m)', 'Y omlaag (m)', 'Z vooruit (m)'], low, high)}
    scene = dict(aspectmode='data', **axes)
    fig.update_layout(height=620, scene=scene, scene2=scene,
                      title=f"Frame {result['report']['frame']} — sleep om de puntenwolk te draaien")
    return fig


def rgb_depth_figure(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(result['rgb']); axes[0].set_title('Opgenomen RGB')
    masked = np.ma.masked_where(result['depth'] <= 0, result['depth'])
    im = axes[1].imshow(masked, cmap='viridis')
    axes[1].set_title('Aligned depth (meters)')
    fig.colorbar(im, ax=axes[1])
    for ax in axes: ax.axis('off')
    fig.tight_layout()
    return fig


def write_ply(path, points):
    """Binary XYZRGB PLY retaining every supplied point, in metres."""
    vertices = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, name in enumerate(('x', 'y', 'z')): vertices[name] = points[:, i]
    for i, name in enumerate(('red', 'green', 'blue')):
        vertices[name] = (points[:, i+3] * 255).round().clip(0, 255).astype(np.uint8)
    header = ('ply\nformat binary_little_endian 1.0\ncomment XYZ in metres\n'
              f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as f:
        f.write(header.encode('ascii')); f.write(vertices.tobytes())


def export_inspection(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{result['report']['frame']:04d}"
    pointcloud_figure(result).write_html(output / f'{stem}.html', include_plotlyjs=True)
    figure = rgb_depth_figure(result)
    figure.savefig(output / f'{stem}_rgb_depth.png', dpi=130)
    import matplotlib.pyplot as plt
    plt.close(figure)
    write_ply(output / f'{stem}_full.ply', result['full'])
    write_ply(output / f'{stem}_sampled.ply', result['sampled'])
    channels = 6 if result['report']['settings']['use_color'] else 3
    np.save(output / f'{stem}_model_points.npy', result['sampled'][:, :channels])
    (output / f'{stem}.json').write_text(json.dumps(result['report'], indent=2) + '\n')
    return output / f'{stem}.html'
