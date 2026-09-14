"""Import a RealSense rs-enumerate-devices text dump without connected hardware."""
import argparse
import hashlib
from pathlib import Path
import re

import h5py
import yaml


def calibration_from_dump(text, camera_name, width, height, aligned):
    serial = re.search(r'^\s*Serial Number\s*:\s*(\S+)', text, re.MULTILINE)
    if serial is None:
        raise ValueError('Camera serial missing in RealSense dump')
    stream = 'Color' if aligned else 'Depth'
    pattern = (rf'Intrinsic of "{stream}" / {width}x{height} /[^\n]*\n'
               r'(.*?)(?=\n\s*Intrinsic of |\nExtrinsic Parameters:|\Z)')
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {stream} profile at {width}x{height}')
    block = matches[0]

    def value(label):
        found = re.search(rf'^\s*{label}:\s*([^\n]+)', block, re.MULTILINE)
        if found is None:
            raise ValueError(f'Missing {label} in {stream} intrinsics')
        return found.group(1).strip()

    distortion = {'None': 'none', 'Brown Conrady': 'brown_conrady',
                  'Inverse Brown Conrady': 'inverse_brown_conrady'}
    model = value('Distortion')
    if model not in distortion:
        raise ValueError(f'Unsupported RealSense distortion: {model}')
    camera = dict(serial=serial.group(1), width=width, height=height, aligned_to_rgb=bool(aligned),
                  intrinsics=dict(fx=float(value('Fx')), fy=float(value('Fy')),
                                  cx=float(value('PPX')), cy=float(value('PPY')),
                                  distortion_model=distortion[model],
                                  coeffs=[float(x) for x in value('Coeffs').split()]),
                  to_reference=None)
    return {'cameras': {camera_name: camera}, 'provenance': {
        'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'intrinsic_stream': stream,
        'reference': f'{camera_name} optical frame: x right, y down, z forward, metres',
        'alignment': 'Color intrinsics for already aligned depth; do not apply depth-to-color again'
                     if aligned else 'Native depth intrinsics',
        'limitation': 'No robot/bottom-camera pose. HDF5 has no serial: camera identity is user-supplied. '
                      'Aligned-depth reconstruction is not recovery of the original native point cloud.',
    }}


def import_dump(dump, episode, output, camera_name='camera_top'):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    with h5py.File(episode, 'r') as root:
        ds = root[f'observations/depth/{camera_name}']
        if 'aligned_to_rgb' not in ds.attrs:
            raise ValueError('HDF5 must explicitly specify aligned_to_rgb')
        height, width = ds.shape[1:]
        result = calibration_from_dump(dump.read_text(), camera_name, width, height,
                                       bool(ds.attrs['aligned_to_rgb']))
    result['provenance']['source_file'] = dump.name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', default='camera_top')
    args = parser.parse_args()
    import_dump(args.dump, args.episode, args.output, args.camera)
    print(f'Wrote {args.output}')
