"""Shared live runner for five-dimensional RGB, RGBD and pointcloud policies."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('J1', 'J2', 'J3', 'J4', 'scissor_roll')


def validate_actions(actions, current, limits, max_delta, check_joint_limits=True, check_delta=True):
    values = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(current, dtype=np.float64)
    maximum = np.asarray(max_delta, dtype=np.float64)
    bounds = np.asarray([limits[name] for name in NAMES], dtype=np.float64)
    if bounds.shape != (5, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('Expected finite ordered lower/upper limits for all five joints')
    if maximum.shape != (5,) or not np.isfinite(maximum).all() or np.any(maximum <= 0):
        raise ValueError('max_delta must contain five positive finite values')
    if previous.shape != (5,) or not np.isfinite(previous).all():
        raise ValueError('Measured state must contain five finite values')
    if values.ndim != 2 or values.shape[1] != 5 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Actions must have finite shape (T,5)')
    for index, row in enumerate(np.vstack([previous, values])):
        outside = (row < bounds[:, 0]) | (row > bounds[:, 1])
        if not check_joint_limits:
            outside[:4] = False
        if outside.any():
            who = 'Measured state' if index == 0 else f'Action {index - 1}'
            details = ', '.join(f'{NAMES[i]}={row[i]:.5f} outside {bounds[i].tolist()}' for i in np.flatnonzero(outside))
            raise ValueError(f'{who}: {details}')
    for index, target in enumerate(values):
        delta = np.abs(target - previous)
        if check_delta and np.any(delta > maximum):
            raise ValueError(f'Action {index} delta {delta.tolist()} exceeds {maximum.tolist()}')
        previous = target
    return values.astype(np.float32)


def limit_action_step(action, current, maximum):
    # Leave rounding margin so the regular delta validator remains authoritative.
    current = np.asarray(current, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64) * 0.999
    return current + np.clip(np.asarray(action, dtype=np.float64) - current, -maximum, maximum)


def read_state(robot, scissor):
    return np.concatenate([robot.get_joint_vector(), [scissor.read()]]).astype(np.float32)


def capture_history(camera, robot, scissor, count, max_skew):
    if hasattr(camera, 'capture_history'):
        return camera.capture_history(count)
    history = []
    for _ in range(count):
        sample = camera.read()
        times = list(sample.get('image_times', {}).values())
        if times and max(times) - min(times) > max_skew:
            raise RuntimeError(f'Camera timestamps differ by {max(times) - min(times):.3f}s')
        history.append((sample, read_state(robot, scissor)))
    return history


def preview_outcome(history):
    import cv2
    from .live import preview_image
    cv2.imshow('Greenhouse: s=success, f=failure, q=stop', preview_image(history[-1][0])[..., ::-1])
    return {ord('s'): 'success', ord('f'): 'failure', ord('q'): 'stopped'}.get(cv2.waitKey(1) & 0xff)


def run_loop(predictor, camera, robot, scissor, settings, args, log):
    limit_steps = getattr(args, 'limit_action_steps', False)
    for index in range(args.chunks):
        history = capture_history(camera, robot, scissor, predictor.n_obs_steps, settings['max_camera_skew_s'])
        started = time.monotonic()
        prediction = predictor.predict(history)
        if hasattr(camera, 'check'):
            camera.check()
        latency = time.monotonic() - started
        end = args.start_action + args.action_count
        if end > len(prediction):
            raise ValueError(f'Requested actions {args.start_action}:{end}, but model returned {len(prediction)}')
        selected = np.array(prediction[args.start_action:end], dtype=np.float64, copy=True)
        # Clip finite policy roll targets, never measured feedback or invalid values.
        # Keep the raw prediction untouched for evaluation in the attempt log.
        if selected.ndim == 2 and selected.shape[1] == 5:
            finite_roll = np.isfinite(selected[:, 4])
            selected[finite_roll, 4] = np.clip(
                selected[finite_roll, 4], *settings['joint_limits']['scissor_roll'])
        current = read_state(robot, scissor)
        error = None
        try:
            validate_actions(selected, current, settings['joint_limits'], settings['max_delta'],
                             settings.get('check_joint_limits', True), check_delta=not limit_steps)
            age = time.monotonic() - history[-1][0].get('image_time', started)
            if age > settings['max_observation_age_s']:
                raise ValueError(f'Observation is {age:.2f}s old after inference')
        except ValueError as exc:
            error = str(exc)
        record = dict(event='prediction', chunk=index, time=time.time(), monotonic_time=time.monotonic(),
                      observation_time=history[-1][0].get('image_time'), qpos=current.tolist(),
                      prediction=prediction.tolist(), selected=selected.tolist(),
                      inference_ms=latency * 1000, execute=args.execute, validation_error=error,
                      limit_action_steps=limit_steps)
        log.write(json.dumps(record) + '\n')
        log.flush()
        print(f'chunk {index + 1}/{args.chunks}: inference={latency * 1000:.1f} ms; '
              f'qpos={np.round(current, 4).tolist()}; target={np.round(selected[0], 4).tolist()}', flush=True)
        if error:
            print('Target check: ' + error, flush=True)
            if args.execute:
                raise ValueError(error)
        if args.show_camera:
            outcome = preview_outcome(history)
            if outcome:
                return outcome
        for action in selected:
            command_start = time.monotonic()
            if args.execute:
                if hasattr(camera, 'check'):
                    camera.check()
                # Recheck measured state before every command, including queued chunk actions.
                measured = read_state(robot, scissor)
                requested_action = action.copy()
                if limit_steps:
                    action = limit_action_step(action, measured, settings['max_delta'])
                validate_actions(action[None], measured, settings['joint_limits'], settings['max_delta'],
                                 settings.get('check_joint_limits', True))
                log.write(json.dumps(dict(event='command_requested', time=time.time(),
                                          monotonic_time=time.monotonic(), target=action.tolist(),
                                          policy_target=requested_action.tolist(), measured=measured.tolist(),
                                          step_limited=bool(np.any(action != requested_action)))) + '\n')
                log.flush()
                scissor.set_rotation(float(action[4]))
                robot.movej(action[:4])
                log.write(json.dumps(dict(event='command', time=time.time(), monotonic_time=time.monotonic(),
                                          target=action.tolist())) + '\n')
                log.flush()
            time.sleep(max(0, 1 / args.hz - (time.monotonic() - command_start)))
            if args.show_camera:
                outcome = preview_outcome(history)
                if outcome:
                    return outcome
    return 'chunk_limit'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('act', 'dp', 'fm', 'act3d', 'dp3', 'fm3'))
    parser.add_argument('modality', choices=('rgb', 'rgbd', 'pointcloud'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/robot_2d.yaml')
    parser.add_argument('--device', default='auto', help='auto (CUDA if available), cpu or cuda:0')
    parser.add_argument('--num-inference-steps', type=int)
    parser.add_argument('--chunks', type=int, default=10)
    parser.add_argument('--start-action', type=int, default=0)
    parser.add_argument('--action-count', type=int, help='Actions per prediction (default: ACT/ACT3D 15, 3D DP/Flow 8, 2D DP/Flow 1)')
    parser.add_argument('--hz', type=float, default=30)
    parser.add_argument('--show-camera', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Enable robot and scissor commands')
    homing = parser.add_mutually_exclusive_group()
    homing.add_argument('--home', dest='home', action='store_true', default=None,
                        help='Home robot 1 during startup (default with --execute); can move the robot')
    homing.add_argument('--skip-home', dest='home', action='store_false',
                        help='Skip homing when robot 1 is already homed and positioned for this attempt')
    parser.add_argument('--prepare-only', action='store_true', help='Initialize control (and optionally home), then exit without policy actions; requires --execute')
    parser.add_argument('--limit-action-steps', action='store_true',
                        help='Clip each target to max_delta from freshly measured state; changes policy execution')
    parser.add_argument('--log-dir', type=Path, default=ROOT / 'runs/robot')
    parser.add_argument('--record-attempt', metavar='NAME', help='Record continuous camera videos and state for this attempt')
    parser.add_argument('--attempt-dir', type=Path, default=ROOT / 'runs/attempts')
    parser.add_argument('--record-depth', action='store_true', help='Also save raw depth for RGBD attempts')
    args = parser.parse_args(argv)
    if args.home and not args.execute:
        parser.error('--home requires --execute')
    if args.home is None:
        args.home = args.execute
    if args.prepare_only and not args.execute:
        parser.error('--prepare-only requires --execute')
    is_3d = args.model in ('act3d', 'dp3', 'fm3')
    if is_3d != (args.modality == 'pointcloud'):
        parser.error('3D models require pointcloud; 2D models require rgb or rgbd')
    if args.action_count is None:
        args.action_count = 15 if args.model in ('act', 'act3d') else (8 if is_3d else 1)
    if args.chunks < 1 or args.action_count < 1 or args.start_action < 0 or not math.isfinite(args.hz) or args.hz <= 0:
        parser.error('Invalid chunks/action selection/hz')
    if args.record_attempt and not re.fullmatch(r'[A-Za-z0-9_-]+', args.record_attempt):
        parser.error('Attempt name may contain letters, numbers, underscores and hyphens')
    if args.record_depth and (not args.record_attempt or args.modality not in ('rgbd', 'pointcloud')):
        parser.error('--record-depth requires --record-attempt with an RGBD or pointcloud model')
    from .inference import Predictor
    from .live import MultiCameraSource
    from .robot import TCSMotionClient, TCSReadClient
    from .scissor import ScissorClient
    settings = yaml.safe_load(args.config.read_text())
    if is_3d:
        from greenhouse_scara_3d_common.inference import Predictor3D
        checkpoint = args.checkpoint or ROOT / f'checkpoints/{args.model}/latest.ckpt'
        predictor = Predictor3D(args.model, checkpoint, args.device, args.num_inference_steps)
    else:
        checkpoint = args.checkpoint or ROOT / f'checkpoints/2d/{args.model}_{args.modality}/current' / ('policy_last.ckpt' if args.model == 'act' else 'latest.ckpt')
        predictor = Predictor(args.model, args.modality, checkpoint, args.device, args.num_inference_steps)
    # Validate configuration before connecting to any hardware.
    midpoint = np.mean([settings['joint_limits'][name] for name in NAMES], axis=1)
    validate_actions(midpoint[None], midpoint, settings['joint_limits'], settings['max_delta'])
    for key in ('max_camera_skew_s', 'max_observation_age_s'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S') + '_' + str(time.time_ns() % 1000000)
    logfile = args.log_dir / f'{args.model}_{args.modality}_{stamp}.jsonl'
    metadata = dict(checkpoint=str(predictor.checkpoint), model=args.model, modality=args.modality,
                    settings=settings, execute=args.execute, device=str(predictor.device),
                    action_count=args.action_count, hz=args.hz, start_action=args.start_action,
                    num_inference_steps=getattr(predictor.policy, 'num_inference_steps', None))
    if is_3d:
        metadata['pointcloud'] = predictor.cfg['pointcloud']
        metadata['calibration'] = predictor.cfg['calibration']
    attempt = None
    if args.record_attempt:
        attempt = args.attempt_dir / f'{stamp}_{args.model}_{args.modality}_{args.record_attempt}'
        attempt.mkdir(parents=True, exist_ok=False)
        logfile = attempt / 'actions.jsonl'
        metadata.update(label=args.record_attempt, started_at=time.time(), outcome='running')
        source = predictor.checkpoint.parent / 'download_source.json'
        if source.is_file():
            metadata['download_source'] = json.loads(source.read_text())
        (attempt / 'attempt.json').write_text(json.dumps(metadata, indent=2) + '\n')
    robot = scissor = camera = None
    recorded_camera = None
    outcome = 'error'
    error = None
    motion_enabled = False
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    previous_signal = signal.signal(signal.SIGTERM, terminate)
    try:
        scissor = ScissorClient(**settings['scissor'])
        scissor.read(wait=True)
        camera = (predictor.camera_source(fps=settings['camera_fps']) if is_3d else
                  MultiCameraSource(predictor.specs, settings['cameras'],
                                    size=tuple(settings['camera_size']), fps=settings['camera_fps']))
        camera.start()
        connection = {k: settings['robot'][k] for k in ('host', 'port', 'timeout')}
        robot = (TCSMotionClient(**settings['robot'], initialize_motion=False)
                 if args.execute else TCSReadClient(**connection))
        if args.execute:
            scissor.prepare_execution()
            robot.enable_motion(home=args.home)
            motion_enabled = True
        current = read_state(robot, scissor)
        if args.execute:
            validate_actions(current[None], current, settings['joint_limits'], settings['max_delta'],
                             settings.get('check_joint_limits', True))
        if args.execute and attempt is not None and not args.prepare_only:
            print('Scissor: moving to 0 rad before recording; waiting for feedback.', flush=True)
            metadata['scissor_zero'] = dict(target=0.0, tolerance_rad=0.02,
                                           started_at=time.time())
            value = scissor.move_to_zero()
            metadata['scissor_zero'].update(measured_rad=value, completed_at=time.time())
            print(f'Scissor ready at {value:.4f} rad; starting attempt recording.', flush=True)
        if attempt is not None:
            from .attempts import AttemptWriter, RecordedCamera
            writer = AttemptWriter(attempt, settings['camera_fps'], args.record_depth)
            recorded_camera = RecordedCamera(camera, lambda: read_state(robot, scissor), writer,
                                             predictor.n_obs_steps, settings['max_camera_skew_s'],
                                             timeout=settings['robot']['timeout'] + 3).start()
        print(f'Mode={"EXECUTE" if args.execute else "READ-ONLY"}; device={predictor.device}; checkpoint={predictor.checkpoint}; log={logfile}', flush=True)
        with logfile.open('x') as log:
            log.write(json.dumps(dict(event='start', **metadata)) + '\n')
            outcome = ('prepared' if args.prepare_only else
                       run_loop(predictor, recorded_camera or camera, robot, scissor, settings, args, log))
    except KeyboardInterrupt:
        outcome = 'interrupted'
        raise
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        # Attempt both stops even if either transport fails.
        if motion_enabled:
            for name, stop in [('robot', robot.halt), ('scissor', scissor.stop)]:
                try:
                    stop()
                except Exception as exc:
                    print(f'FAILED to stop {name}: {exc}', file=sys.stderr)
                    metadata.setdefault('stop_errors', []).append(f'{name}: {exc}')
        if recorded_camera is not None:
            try:
                recorded_camera.close()
            except Exception as exc:
                error = error or str(exc)
                outcome = 'error'
                print(f'Attempt recording failed: {exc}', file=sys.stderr)
        for resource in (robot, camera, scissor):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    print(f'Cleanup failed: {exc}', file=sys.stderr)
        if args.show_camera:
            import cv2
            cv2.destroyAllWindows()
        signal.signal(signal.SIGTERM, previous_signal)
        if attempt is not None:
            metadata.update(outcome=outcome, error=error, finished_at=time.time())
            if recorded_camera is not None:
                metadata.update(recorded_camera.writer.summary())
            temporary = attempt / 'attempt.json.tmp'
            temporary.write_text(json.dumps(metadata, indent=2) + '\n')
            os.replace(temporary, attempt / 'attempt.json')
            print(f'Attempt saved: {attempt} (outcome={outcome})', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped by user/signal.', file=sys.stderr)
