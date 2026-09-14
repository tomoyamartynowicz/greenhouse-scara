"""Read the actual depth-frame intrinsics after optional RGB alignment; no robot connection."""
import argparse
from pathlib import Path
import yaml


def calibration_from_frame(frame, serial, aligned):
    intr = frame.profile.as_video_stream_profile().get_intrinsics()
    return dict(serial=str(serial), width=intr.width, height=intr.height, aligned_to_rgb=aligned,
                intrinsics=dict(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                                distortion_model=str(intr.model).split('.')[-1], coeffs=list(intr.coeffs)),
                to_reference=None)


def capture(name, serial, aligned):
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    try:
        align = rs.align(rs.stream.color) if aligned else None
        for _ in range(10): frames = pipeline.wait_for_frames(10000)
        if align: frames = align.process(frames)
        frame = frames.get_depth_frame()
        if not frame: raise RuntimeError(f'No depth frame from {name}')
        return calibration_from_frame(frame, serial, aligned)
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', nargs=2, action='append', metavar=('NAME', 'SERIAL'))
    parser.add_argument('--raw-depth', action='store_true')
    parser.add_argument('--output', default=str(Path(__file__).with_name('calibration.yaml')))
    args = parser.parse_args()
    output = Path(args.output)
    cameras = args.camera or [('camera_top', '130322273198')]
    if len({name for name, _ in cameras}) != len(cameras): raise ValueError('Duplicate camera name')
    # Never replace measured calibration/extrinsics implicitly.
    if output.exists():
        current = yaml.safe_load(output.read_text()) or {}
        if any(c.get('intrinsics', {}).get('fx') is not None for c in current.get('cameras', {}).values()):
            raise FileExistsError(f'{output} contains calibration; choose a new --output path')
    values = {'cameras': {name: capture(name, serial, not args.raw_depth) for name, serial in cameras}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(values, sort_keys=False))
    print(f'Calibration saved to {output}. For multiple cameras, set to_reference from an extrinsic calibration.')


if __name__ == '__main__': main()
