"""Multiple RealSense inputs with the same preprocessing used by the HDF5 loaders."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .data import preprocess_image

DEFAULT_SERIALS = {'camera_top': '130322273198', 'camera_bottom': '123622270497'}


class MultiCameraSource:
    def __init__(self, specs, serials=None, size=(640, 480), fps=30):
        from .realsense import RealSenseSource
        self.specs = specs
        serials = dict(DEFAULT_SERIALS if serials is None else serials)
        names = list(dict.fromkeys(spec.get('camera', key) for key, spec in specs.items()))
        if any(not serials.get(name) for name in names):
            raise ValueError(f'Provide a serial for every physical camera: {names}')
        if len({serials[name] for name in names}) != len(names):
            raise ValueError('Each physical camera needs a different serial')
        self.sources = []
        for name in names:
            depth_specs = [s for k,s in specs.items() if s.get('camera', k) == name and s['type'] == 'depth']
            aligned = {bool(s.get('aligned_to_rgb', False)) for s in depth_specs}
            if len(aligned) > 1: raise ValueError(f'Conflicting depth alignment for {name}')
            self.sources.append(RealSenseSource(name, serials[name], size=size, fps=fps,
                                               depth=bool(depth_specs), align_depth=next(iter(aligned), False)))
        self.executor = ThreadPoolExecutor(max_workers=len(self.sources))

    def start(self):
        try:
            for source in self.sources: source.start()
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        futures = [self.executor.submit(source.read) for source in self.sources]
        samples = [future.result() for future in futures]
        packet = {'frames': {}, 'depth_frames': {}, 'depth_scales': {}}
        for sample in samples:
            packet['frames'].update(sample['frames'])
            packet['depth_frames'].update(sample.get('depth_frames', {}))
            for name in sample.get('depth_frames', {}):
                packet['depth_scales'][name] = sample['depth_scale']
        return packet

    def close(self):
        self.executor.shutdown(wait=True)
        error = None
        for source in reversed(self.sources):
            try: source.close()
            except Exception as exc: error = exc
        if error is not None: raise error


def observation_from_history(history, specs, device):
    result = {}
    for key,spec in specs.items():
        name = spec.get('camera', key)
        group = 'frames' if spec['type'] == 'rgb' else 'depth_frames'
        images = [preprocess_image(sample[group][name], spec, sample['depth_scales'].get(name))
                  for sample, _ in history]
        result[key] = torch.from_numpy(np.stack(images))[None].to(device)
    result['qpos'] = torch.from_numpy(np.stack([qpos for _,qpos in history]).astype(np.float32))[None].to(device)
    return result


def preview_image(sample):
    return np.concatenate(list(sample['frames'].values()), axis=1)
