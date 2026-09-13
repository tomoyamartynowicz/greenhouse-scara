from __future__ import annotations

import math
import telnetlib
import threading
import time
from typing import Mapping

import numpy as np

from constants import ACT_STATE_DIM, DEFAULT_JOINT_TARGET, JOINT_LIMITS, JOINT_NAMES


class TCSCommandError(RuntimeError):
    def __init__(self, command: str, reply: str) -> None:
        super().__init__(f"TCS error on {command!r}: {reply}")
        self.command = command
        self.reply = reply


class JointLimitError(ValueError):
    pass


def complete_joint_target(target: Mapping[str, float]) -> dict[str, float]:
    complete = dict(DEFAULT_JOINT_TARGET)
    complete.update(target)
    return {joint: float(complete[joint]) for joint in JOINT_NAMES}


def require_joint_target_within_limits(target: Mapping[str, float]) -> dict[str, float]:
    complete = complete_joint_target(target)
    violations = []
    for joint in JOINT_NAMES:
        value = float(complete[joint])
        lower, upper = JOINT_LIMITS[joint]
        if value < lower or value > upper:
            violations.append(f"{joint}={value:.6g} outside [{lower:.6g}, {upper:.6g}]")
    if violations:
        raise JointLimitError("; ".join(violations))
    return complete


def joint_dict_to_vector(target: Mapping[str, float]) -> np.ndarray:
    complete = complete_joint_target(target)
    vector = np.zeros(ACT_STATE_DIM, dtype=np.float32)
    for index, joint in enumerate(JOINT_NAMES):
        vector[index] = float(complete[joint])
    return vector


def vector_to_joint_dict(vector: np.ndarray | list[float]) -> dict[str, float]:
    if len(vector) < len(JOINT_NAMES):
        raise ValueError(f"expected at least {len(JOINT_NAMES)} joint values, got {vector!r}")
    return {joint: float(vector[index]) for index, joint in enumerate(JOINT_NAMES)}


def format_joint_target(target: Mapping[str, float]) -> str:
    complete = complete_joint_target(target)
    return " ".join(f"{joint}={float(complete[joint]):.4f}" for joint in JOINT_NAMES)


class TCSBaseClient:
    def __init__(self, host: str = "192.168.0.10", port: int = 10100, timeout: float = 5.0, verbose: bool = False) -> None:
        self.host = host
        self.port = int(port)
        self.verbose = bool(verbose)
        self.lock = threading.Lock()
        self.connection = telnetlib.Telnet(host, port, timeout)

    def command(self, command: str) -> str:
        if self.verbose:
            print(f"TCS command: {command}")

        with self.lock:
            self.connection.write((command + "\n").encode("ascii"))
            line = self.connection.read_until(b"\n").decode("ascii").strip()

        parts = line.split(" ", 1)
        code = parts[0]
        data = parts[1] if len(parts) > 1 else ""

        if code.startswith("-"):
            raise TCSCommandError(command, line)

        return data

    def command_sleep(self, command: str, delay: float = 0.15) -> str:
        reply = self.command(command)
        time.sleep(max(0.0, float(delay)))
        return reply

    def close(self) -> None:
        self.connection.close()


class TCSReadClient(TCSBaseClient):
    def __init__(
        self, host: str = "192.168.0.10", port: int = 10100, timeout: float = 5.0, verbose: bool = False) -> None:
        super().__init__(host=host, port=port, timeout=timeout, verbose=verbose)

    def get_wherej(self) -> list[float]:
        reply = self.command("wherej")
        return [float(value) for value in reply.split()]

    def get_joint_state(self) -> dict[str, float]:
        values = self.get_wherej()
        if len(values) < 4:
            raise ValueError(f"expected at least 4 joint values from wherej, got {values!r}")
        return {"J1": values[0] * 0.001,
                "J2": values[1] * math.pi/180.0,
                "J3": values[2] * math.pi/180.0,
                "J4": values[3] * math.pi/180.0,}


class TCSMotionClient(TCSReadClient):
    def __init__(
        self, 
        host: str = "192.168.0.10", port: int = 10100, timeout: float = 5.0,
        verbose: bool = False, profile: int = 2, mspeed: int = 100, 
        profile_speed: int = 50, profile_accel: int = 50,
        profile_ramp: float = 0.08, profile_straight: int = 0,
        configure_motion: bool = True, set_tool: bool = True) -> None:
        super().__init__(host=host, port=port, timeout=timeout, verbose=verbose)
        self.profile = int(profile)
        self.axis_count = 4
        self.startup()
        if configure_motion:
            self.configure_motion(mspeed=mspeed, profile_speed=profile_speed, profile_accel=profile_accel, profile_ramp=profile_ramp, profile_straight=profile_straight, set_tool=set_tool)

    def startup(self) -> None:
        startup_delay = 0.5
        query_delay = 0.10
        self.command_sleep("SelectRobot 1", startup_delay)
        for command in ("attach", "hp", "wherec", "state"):
            try:
                self.command_sleep(command, startup_delay)
            except TCSCommandError as exc:
                self.warn_startup_failure(command, exc)
        try:
            values = self.get_wherej()
            self.axis_count = len(values)
        except (TCSCommandError, ValueError) as exc:
            if isinstance(exc, TCSCommandError):
                self.warn_startup_failure("wherej", exc)

        self.ensure_motion_ready()

    def warn_startup_failure(self, command: str, exc: TCSCommandError) -> None:
        if self.verbose:
            print(f"TCS startup warning: {command!r} failed with {exc.reply!r}; continuing.")

    def ensure_motion_ready(self, commands: list[str] | tuple[str, ...] = ("attach 1", "hp 1 -1"), delay: float = 0.5) -> list[tuple[str, str]]:
        failures = []
        for command in commands:
            try:
                self.command_sleep(command, delay)
            except TCSCommandError as exc:
                failures.append((command, exc.reply))
        return failures

    def configure_motion(self, mspeed: int = 20, profile_speed: int = 35, 
                         profile_accel: int = 35, profile_ramp: float = 0.08,
                         profile_straight: int = 0, set_tool: bool = True,) -> None:
        self.command(f"mspeed {int(mspeed)}")
        self.command(f"profile {self.profile} {int(profile_speed)} {int(profile_speed)} "
                     f"{int(profile_accel)} {int(profile_accel)} "
                     f"{float(profile_ramp)} {float(profile_ramp)} -1 {int(profile_straight)}")
        if set_tool:
            self.command("tool 0 0 0 0 0 0")

    def movej(self, joints: dict[str, float]) -> None:
        joints = require_joint_target_within_limits(joints)
        values = [joints["J1"] * 1000.0, joints["J2"] * 180.0 / math.pi,
                  joints["J3"] * 180.0 / math.pi, joints["J4"] * 180.0 / math.pi,]

        values_text = " ".join(f"{value:.6g}" for value in values)
        self.command(f"MoveJ {self.profile} {values_text}")

    def halt(self) -> None:
        self.command("halt")
