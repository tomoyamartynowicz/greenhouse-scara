#!/usr/bin/env python3

import argparse
import pickle
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenhouse_scara_common.live import MultiCameraSource, observation_from_history, preview_image, DEFAULT_SERIALS
from policy import ACTPolicy
from tcs_client import TCSMotionClient, TCSReadClient, joint_dict_to_vector, vector_to_joint_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="policy_best.ckpt")
    parser.add_argument("--chunks", type=int, default=1000, help="number of observations/predicted chunks")
    parser.add_argument("--start-action", type=int, default=0, help="first action index sent from every chunk")
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="192.168.0.10")
    parser.add_argument("--port", type=int, default=10100)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--camera", action="append", nargs=2, metavar=("NAME", "SERIAL"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-camera-view", action="store_true")
    parser.add_argument("--profile", type=int, default=2)
    parser.add_argument("--mspeed", type=int, default=100)
    parser.add_argument("--profile-speed", type=int, default=20)
    parser.add_argument("--profile-accel", type=int, default=20)
    parser.add_argument("--profile-ramp", type=float, default=0.08)
    parser.add_argument("--profile-straight", type=int, default=0)
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    with open(ckpt_dir / "config.pkl", "rb") as f: config = pickle.load(f)
    with open(ckpt_dir / "dataset_stats.pkl", "rb") as f: stats = pickle.load(f)

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute(): checkpoint = ckpt_dir / checkpoint
    state_dict = torch.load(checkpoint, map_location="cpu")
    if "policy" in state_dict: state_dict = state_dict["policy"]
    config["policy_config"]["num_queries"] = state_dict["model.query_embed.weight"].shape[0]

    device = torch.device(args.device)
    if device.index is not None: torch.cuda.set_device(device)
    config["policy_config"]["pretrained_backbone"] = False  # Weights come from the checkpoint.
    builder_args = ["eval", "--ckpt_dir", str(ckpt_dir), "--policy_class", "ACT", "--task_name", config["task_name"], "--seed", str(config["seed"]), "--num_epochs", str(config["num_epochs"])]
    with patch.object(sys, "argv", builder_args): policy = ACTPolicy(config["policy_config"])
    policy.load_state_dict(state_dict)
    policy.to(device).eval()

    qpos_mean, qpos_std = stats["qpos_mean"], stats["qpos_std"]
    action_mean, action_std = stats["action_mean"], stats["action_std"]
    specs = config['policy_config']['stream_specs']
    serials = dict(DEFAULT_SERIALS)
    if args.camera: serials.update(dict(args.camera))
    if args.serial:
        names = {spec.get('camera', key) for key,spec in specs.items()}
        if len(names) != 1: raise ValueError('Use --camera NAME SERIAL for multiple cameras')
        serials[next(iter(names))] = args.serial
    camera = MultiCameraSource(specs, serials)
    robot = None
    viewer = None

    try:
        camera.start()
        if not args.no_viewer:
            from live_viewer import LiveViewer
            viewer = LiveViewer(show_camera=not args.no_camera_view)
        if args.execute:
            robot = TCSMotionClient(host=args.host, port=args.port, profile=args.profile, mspeed=args.mspeed, profile_speed=args.profile_speed, profile_accel=args.profile_accel, profile_ramp=args.profile_ramp, profile_straight=args.profile_straight, set_tool=False)
        else:
            robot = TCSReadClient(host=args.host, port=args.port)

        for chunk_number in range(args.chunks):
            sample = camera.read()
            camera_image = preview_image(sample)
            current_q = joint_dict_to_vector(robot.get_joint_state())
            observation = observation_from_history([(sample, current_q)], specs, device)
            image = {key: observation[key][:, 0] for key in specs}
            qpos = torch.from_numpy((current_q - qpos_mean) / qpos_std)[None].float().to(device)

            with torch.inference_mode(): actions = policy(qpos, image)[0].cpu().numpy()
            actions = actions * action_std + action_mean
            actions = actions[args.start_action:]
            if viewer is not None: viewer.update(camera_image, current_q, actions, chunk_number + 1)
            print(f"chunk {chunk_number + 1}/{args.chunks}: sending {len(actions)} actions from index {args.start_action} at {args.hz:g} Hz")

            for action in actions:
                command_start = time.monotonic()
                if args.execute: robot.movej(vector_to_joint_dict(action))
                else: print(np.round(action, 4))
                time.sleep(max(0, 1 / args.hz - (time.monotonic() - command_start)))

    finally:
        if args.execute and robot is not None:
            try: robot.halt()
            except Exception: pass
        if viewer is not None: viewer.close()
        camera.close()
        if robot is not None: robot.close()


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nstopped")
