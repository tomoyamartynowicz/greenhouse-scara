"""One RGB/depth camera class, instantiated for each configured RealSense."""

import time

import numpy as np
import pyrealsense2 as rs


class RealSenseSource:
    def __init__(self, color_name, serial, size=(640, 480), fps=30,
                 depth=False, align_depth=False):
        self.color_name = color_name
        self.serial = serial
        self.size = size
        self.fps = fps
        self.depth = depth
        self.align_depth = depth and align_depth
        self.depth_scale = None
        self.align = None
        self.pipeline = None

    def start(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, *self.size, rs.format.rgb8, self.fps)
        if self.depth:
            config.enable_stream(rs.stream.depth, *self.size, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        self.pipeline = pipeline
        try:
            if self.depth:
                self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            if self.align_depth:
                self.align = rs.align(rs.stream.color)
            for _ in range(30):
                self.pipeline.wait_for_frames(1000)
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        frameset = self.pipeline.wait_for_frames(1000)
        # Same host clock as the joint poller, sampled immediately after arrival.
        image_time = time.monotonic()
        if self.align is not None:
            frameset = self.align.process(frameset)
        color_frame = frameset.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"Missing RGB frame from {self.color_name}")
        sample = {
            "frames": {self.color_name: np.asanyarray(color_frame.get_data()).copy()},
            "image_time": image_time,
        }
        if self.depth:
            depth_frame = frameset.get_depth_frame()
            if not depth_frame:
                raise RuntimeError(f"Missing depth frame from {self.color_name}")
            sample["depth_frames"] = {
                self.color_name: np.asanyarray(depth_frame.get_data()).copy(),
            }
            sample["depth_scale"] = self.depth_scale
            sample["depth_aligned"] = self.align_depth
        return sample

    def close(self):
        if self.pipeline is not None:
            pipeline = self.pipeline
            self.pipeline = None
            pipeline.stop()
