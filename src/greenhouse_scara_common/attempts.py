"""Continuous attempt recording using the same camera streams as inference."""
from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import threading
import time
import numpy as np


class AttemptWriter:
    def __init__(self, directory, fps, record_depth=False):
        self.directory = Path(directory)
        self.fps = fps
        self.record_depth = record_depth
        self.videos = {}
        self.depth_file = None
        self.telemetry = (self.directory / 'observations.jsonl').open('x')
        self.frames = 0
        self.first_time = self.last_time = None

    def append(self, sample, qpos, state_time):
        import cv2
        for name, rgb in sample['frames'].items():
            if name not in self.videos:
                height, width = rgb.shape[:2]
                writer = cv2.VideoWriter(str(self.directory / f'{name}.avi'),
                                         cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (width, height))
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f'Cannot create video for {name}')
                self.videos[name] = writer
            self.videos[name].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if self.record_depth:
            import h5py
            if not sample.get('depth_frames'):
                raise ValueError('Depth recording requested but no depth frames received')
            if self.depth_file is None:
                self.depth_file = h5py.File(self.directory / 'depth.hdf5', 'x')
            for name, depth in sample['depth_frames'].items():
                if name not in self.depth_file:
                    ds = self.depth_file.create_dataset(name, shape=(0, *depth.shape),
                        maxshape=(None, *depth.shape), dtype=np.uint16,
                        chunks=(1, *depth.shape), compression='lzf')
                    ds.attrs['depth_scale'] = sample['depth_scales'][name]
                    ds.attrs['aligned_to_rgb'] = bool(sample.get('depth_aligned', {}).get(name, False))
                ds = self.depth_file[name]
                ds.resize(self.frames + 1, axis=0)
                ds[self.frames] = depth
            self.depth_file.flush()
        timestamp = sample['image_time']
        record = dict(frame=self.frames, image_time=timestamp,
                      image_times=sample.get('image_times', {}), state_time=state_time,
                      qpos=np.asarray(qpos).tolist(), wall_time=time.time())
        self.telemetry.write(json.dumps(record, allow_nan=False) + '\n')
        self.telemetry.flush()
        self.frames += 1
        self.first_time = timestamp if self.first_time is None else self.first_time
        self.last_time = timestamp

    def summary(self):
        duration = (self.last_time - self.first_time) if self.frames > 1 else 0
        return dict(recorded_frames=self.frames, recorded_duration_s=duration,
                    measured_fps=(self.frames - 1) / duration if duration > 0 else 0,
                    video_playback_fps=self.fps, depth_recorded=self.record_depth)

    def close(self):
        for writer in self.videos.values():
            writer.release()
        if self.depth_file is not None:
            self.depth_file.close()
        self.telemetry.close()


class RecordedCamera:
    """One camera reader feeds a bounded inference history and the recording.

    State sampling is timestamped after each paired camera read, without the
    demonstrator's interpolation. This is an evaluation log, not a new demo.
    """
    def __init__(self, camera, read_state, writer, history_size, max_skew, timeout=5.0):
        self.camera, self.read_state, self.writer = camera, read_state, writer
        self.max_skew, self.timeout = max_skew, timeout
        self.history = deque(maxlen=max(2, history_size))
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.error = None
        self.sequence = self.consumed = 0
        self.thread = threading.Thread(target=self._capture, name='attempt-recording', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _capture(self):
        try:
            while not self.stopping.is_set():
                sample = self.camera.read()
                times = list(sample.get('image_times', {}).values())
                if times and max(times) - min(times) > self.max_skew:
                    raise RuntimeError('Camera timestamps differ too much during recording')
                qpos = self.read_state()
                self.writer.append(sample, qpos, time.monotonic())
                with self.condition:
                    self.history.append((sample, qpos))
                    self.sequence += 1
                    self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            try:
                self.writer.close()
            except BaseException as exc:
                with self.condition:
                    self.error = self.error or exc
                    self.condition.notify_all()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Attempt recording stopped: ' + str(self.error)) from self.error

    def capture_history(self, count):
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while len(self.history) < count or self.sequence <= self.consumed:
                self.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.stopping.is_set():
                    raise TimeoutError('No fresh recorded camera observations')
                self.condition.wait(remaining)
            self.check()
            self.consumed = self.sequence
            return list(self.history)[-count:]

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=self.timeout)
        if self.thread.is_alive():
            raise TimeoutError('Attempt recorder did not finish closing')
        self.check()
