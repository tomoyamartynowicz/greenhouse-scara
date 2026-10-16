"""ROS2 scissor feedback/commands, matching scara_dataset's radian interface."""
from __future__ import annotations
import math
import threading
import time


class ScissorClient:
    def __init__(self, topic='/scissor/rotation', service='/scissor/set_rotation',
                 halt_service='/scissor/halt', timeout=1.0, max_state_age_s=0.3,
                 stop_mode='auto', command_epsilon_rad=0.005):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Float64
        from std_srvs.srv import Trigger
        from scissor_interfaces.srv import SetRotation
        if stop_mode not in ('auto', 'halt', 'hold'):
            raise ValueError('scissor.stop_mode must be auto, halt or hold')
        if timeout <= 0 or max_state_age_s <= 0 or command_epsilon_rad < 0:
            raise ValueError('Invalid scissor timeouts/epsilon')
        self.rclpy, self.rotation_type, self.trigger_type = rclpy, SetRotation, Trigger
        self.timeout, self.max_age = float(timeout), float(max_state_age_s)
        self.stop_mode, self.epsilon = stop_mode, float(command_epsilon_rad)
        self.condition = threading.Condition()
        self.sample = None
        self.error = None
        self.last_target = None
        self.closed = False
        self.owns_ros = not rclpy.ok()
        if self.owns_ros:
            rclpy.init()
        self.node = rclpy.create_node('greenhouse_policy_scissor')
        self.subscription = self.node.create_subscription(Float64, topic, self._receive, 10)
        self.service = self.node.create_client(SetRotation, service)
        self.halt_service = self.node.create_client(Trigger, halt_service)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _receive(self, msg):
        with self.condition:
            self.sample = (float(msg.data), time.monotonic())
            self.condition.notify_all()

    def _spin(self):
        try:
            self.executor.spin()
        except BaseException as exc:
            if not self.closed:
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()

    def read(self, wait=False):
        deadline = time.monotonic() + (self.timeout if wait else 0)
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError('Scissor ROS executor failed') from self.error
                if self.sample is not None:
                    value, timestamp = self.sample
                    if not math.isfinite(value):
                        raise ValueError('Scissor feedback is not finite')
                    if time.monotonic() - timestamp <= self.max_age:
                        return value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Missing/stale /scissor/rotation feedback')
                self.condition.wait(remaining)

    def prepare_execution(self):
        if not self.service.wait_for_service(timeout_sec=self.timeout):
            raise TimeoutError('/scissor/set_rotation is unavailable')
        has_halt = self.halt_service.wait_for_service(timeout_sec=0.5)
        if self.stop_mode == 'halt' and not has_halt:
            raise TimeoutError('/scissor/halt is required by scissor.stop_mode=halt')
        self.use_halt = has_halt and self.stop_mode != 'hold'
        if not self.use_halt:
            print('Scissor stop: hold measured position (halt service unavailable or hold selected).', flush=True)
        return self.read(wait=True)

    def _call(self, client, request):
        if not client.wait_for_service(timeout_sec=self.timeout):
            raise TimeoutError('Scissor service unavailable')
        future = client.call_async(request)
        deadline = time.monotonic() + self.timeout
        while not future.done():
            if self.error is not None:
                raise RuntimeError('Scissor executor failed') from self.error
            if time.monotonic() >= deadline:
                future.cancel()
                raise TimeoutError('Scissor command timed out; acceptance is unknown')
            time.sleep(0.002)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError('Scissor command rejected: ' + str(getattr(response, 'message', response)))

    def set_rotation(self, value, force=False):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Scissor target is not finite')
        if not force and self.last_target is not None and abs(value - self.last_target) < self.epsilon:
            return
        request = self.rotation_type.Request()
        request.rotation = value
        self._call(self.service, request)
        self.last_target = value

    def stop(self):
        if getattr(self, 'use_halt', False):
            self._call(self.halt_service, self.trigger_type.Request())
        else:
            # This is a position hold, not a hardware emergency stop.
            self.set_rotation(self.read(), force=True)

    def move_to_zero(self, timeout=10.0, tolerance=0.02, settle_time=0.3):
        """Command zero once and require sustained, fresh feedback before returning."""
        if not all(math.isfinite(x) and x > 0 for x in (timeout, tolerance, settle_time)):
            raise ValueError('Invalid scissor zeroing parameters')
        self.read(wait=True)
        self.set_rotation(0.0, force=True)
        deadline = time.monotonic() + timeout
        settled_since = None
        while time.monotonic() < deadline:
            value = self.read()  # Missing/stale feedback must abort, not count as zero.
            now = time.monotonic()
            if abs(value) <= tolerance:
                if settled_since is None:
                    settled_since = now
                if now - settled_since >= settle_time:
                    return value
            else:
                settled_since = None
            time.sleep(0.02)
        raise TimeoutError('Scissor did not reach zero before recording; model execution cancelled')

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(timeout_sec=self.timeout)
        self.thread.join(timeout=self.timeout)
        self.node.destroy_node()
        if self.owns_ros and self.rclpy.ok():
            self.rclpy.shutdown()
