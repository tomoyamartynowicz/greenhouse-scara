"""TCS transport adapted from the existing DP live runner; units are m/rad."""
from __future__ import annotations
import math
import telnetlib
import threading
import time
from typing import Sequence
import numpy as np

JOINT_NAMES = ("J1", "J2", "J3", "J4")
class TCSCommandError(RuntimeError):
    def __init__(self, command: str, reply: str) -> None:
        hints = {'-1046': 'Motor power is not enabled.',
                 '-3101': 'PID output saturated too long: inspect the controller error log and robot before restarting; this is not a missing-home indication.',
                 '-1047': 'Virtual MCP is in Jog mode; switch to Computer mode.',
                 '-1048': 'Hardware MCP is in Jog mode; switch to Computer mode.'}
        hint = hints.get(reply.split()[0] if reply.split() else '', '')
        super().__init__(f"TCS error on {command!r}: {reply}" + (f' ({hint})' if hint else ''))
        self.command = command
        self.reply = reply


class TCSBaseClient:
    def __init__(
        self,
        host: str = "192.168.1.10",
        port: int = 10100,
        timeout: float = 5.0,
        verbose: bool = False,
    ) -> None:
        self.verbose = bool(verbose)
        self.timeout = float(timeout)
        self.lock = threading.Lock()
        self.connection = telnetlib.Telnet(host, int(port), float(timeout))

    def command(self, command: str, response_timeout: float | None = None) -> str:
        if self.verbose:
            print(f"TCS command: {command}", flush=True)
        with self.lock:
            self.connection.write((command + "\n").encode("ascii"))
            raw = self.connection.read_until(b"\n", self.timeout if response_timeout is None else response_timeout)
            if not raw.endswith(b"\n"):
                raise TCSCommandError(command, "incomplete reply or timeout")
            line = raw.decode("ascii").strip()
        if not line:
            raise TCSCommandError(command, "empty reply")
        parts = line.split(" ", 1)
        if parts[0].startswith("-"):
            raise TCSCommandError(command, line)
        return parts[1] if len(parts) > 1 else ""

    def command_sleep(self, command: str, delay: float = 0.15) -> str:
        reply = self.command(command)
        time.sleep(max(0.0, float(delay)))
        return reply

    def close(self) -> None:
        self.connection.close()


class TCSReadClient(TCSBaseClient):
    def get_joint_vector(self) -> np.ndarray:
        values = [float(value) for value in self.command("wherej").split()]
        if len(values) < len(JOINT_NAMES):
            raise ValueError(f"Expected four values from wherej, got {values!r}")
        return np.asarray(
            [
                values[0] * 0.001,
                values[1] * math.pi / 180.0,
                values[2] * math.pi / 180.0,
                values[3] * math.pi / 180.0,
            ],
            dtype=np.float32,
        )


class TCSMotionClient(TCSReadClient):
    def __init__(
        self,
        host: str = "192.168.1.10",
        port: int = 10100,
        timeout: float = 5.0,
        verbose: bool = False,
        profile: int = 2,
        mspeed: int = 100,
        profile_speed: int = 20,
        profile_accel: int = 20,
        profile_ramp: float = 0.08,
        profile_straight: int = 0,
        initialize_motion: bool = True,
    ) -> None:
        super().__init__(host=host, port=port, timeout=timeout, verbose=verbose)
        self.profile = int(profile)
        self.motion_settings = dict(
            mspeed=mspeed, profile_speed=profile_speed, profile_accel=profile_accel,
            profile_ramp=profile_ramp, profile_straight=profile_straight,
        )
        if initialize_motion:
            self.enable_motion()

    def enable_motion(self, home: bool = False) -> None:
        """Enable control on the existing connection after the caller's checks."""
        try:
            self._startup()
            if home:
                print('TCS: homing robot 1; the robot may move.', flush=True)
                self.command('home', response_timeout=120.0)
                self.get_joint_vector()
                print('TCS: home command completed.', flush=True)
            self._configure_motion(**self.motion_settings)
        except BaseException:
            # Startup/homing may already have enabled motion. Attempt a stop
            # before releasing this connection; never reset/retry controller faults.
            try:
                self.halt()
            except Exception:
                pass
            self.close()
            raise

    def _startup(self) -> None:
        self.command_sleep("SelectRobot 1")
        self.command("state")  # Surface outstanding controller faults before enabling power.
        # TCS requires power before attaching. Keep replies synchronous and
        # allow the controller's finite power-enable wait to finish.
        if self.command("hp") != "1":
            print('TCS: enabling motor power (up to 30 seconds).', flush=True)
            self.command("hp 1 30", response_timeout=35.0)
        if self.command_sleep("hp") != "1":
            raise TCSCommandError("hp", "high power is not enabled")
        if self.command("attach") != "1":
            self.command_sleep("attach 1")
        if self.command("attach") != "1":
            raise TCSCommandError("attach", "robot 1 is not attached")
        print('TCS: robot 1 selected, motor power on, attached.', flush=True)

    def _configure_motion(
        self,
        mspeed: int,
        profile_speed: int,
        profile_accel: int,
        profile_ramp: float,
        profile_straight: int,
    ) -> None:
        self.command(f"mspeed {int(mspeed)}")
        self.command(
            f"profile {self.profile} {int(profile_speed)} {int(profile_speed)} "
            f"{int(profile_accel)} {int(profile_accel)} "
            f"{float(profile_ramp)} {float(profile_ramp)} -1 {int(profile_straight)}"
        )

    def movej(self, target: Sequence[float]) -> None:
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (4,) or not np.isfinite(target).all():
            raise ValueError("MoveJ requires four finite values")
        values = (
            target[0] * 1000.0,
            target[1] * 180.0 / math.pi,
            target[2] * 180.0 / math.pi,
            target[3] * 180.0 / math.pi,
        )
        self.command(
            f"MoveJ {self.profile} " + " ".join(f"{value:.6g}" for value in values)
        )

    def halt(self) -> None:
        self.command("halt")

