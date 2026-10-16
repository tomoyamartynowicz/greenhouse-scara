import importlib.util
import io
import json
import pickle
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from greenhouse_scara_common.deployment import validate_actions, run_loop, NAMES

spec = importlib.util.spec_from_file_location('fetch_2d', ROOT / 'scripts/fetch_2d_checkpoints.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class ScissorZeroTests(unittest.TestCase):
    def run_zero(self, readings, timeout=1.0):
        from greenhouse_scara_common.scissor import ScissorClient
        from unittest.mock import Mock
        clock = [0.0]
        values = iter(readings)
        last = [readings[-1]]
        def read(**kwargs):
            last[0] = next(values, last[0])
            if isinstance(last[0], Exception):
                raise last[0]
            return last[0]
        def sleep(seconds): clock[0] += seconds
        client = SimpleNamespace(read=read, set_rotation=Mock())
        with patch('greenhouse_scara_common.scissor.time.monotonic', side_effect=lambda: clock[0]), \
             patch('greenhouse_scara_common.scissor.time.sleep', side_effect=sleep):
            result = ScissorClient.move_to_zero(client, timeout=timeout)
        client.set_rotation.assert_called_once_with(0.0, force=True)
        return result, clock[0]

    def test_waits_for_zero_to_settle_after_transient(self):
        value, elapsed = self.run_zero([0.8, 0, 0.2, 0.01])
        self.assertEqual(value, 0.01)
        self.assertGreaterEqual(elapsed, 0.34)

    def test_timeout_and_stale_feedback_abort(self):
        with self.assertRaisesRegex(TimeoutError, 'did not reach zero'):
            self.run_zero([0.8], timeout=0.1)
        with self.assertRaisesRegex(TimeoutError, 'stale'):
            self.run_zero([0.8, TimeoutError('stale')])


class TCSConnectionTests(unittest.TestCase):
    def test_startup_power_before_attach_and_optional_home(self):
        from greenhouse_scara_common.robot import TCSMotionClient
        for home in (False, True):
            state = {'hp': False, 'attach': False}
            commands = []
            def command(text, response_timeout=None):
                commands.append(text)
                if text == 'hp': return str(int(state['hp']))
                if text == 'hp 1 30':
                    self.assertEqual(response_timeout, 35.0)
                    state['hp'] = True
                if text == 'attach': return str(int(state['attach']))
                if text == 'attach 1':
                    self.assertTrue(state['hp'])
                    state['attach'] = True
                if text == 'home':
                    self.assertTrue(state['attach'])
                    self.assertEqual(response_timeout, 120.0)
                if text == 'wherej': return '100 0 0 0'
                return '0'
            with patch('greenhouse_scara_common.robot.telnetlib.Telnet'), \
                 patch('greenhouse_scara_common.robot.time.sleep'):
                robot = TCSMotionClient(initialize_motion=False)
                robot.command = command
                robot.enable_motion(home=home)
                self.assertEqual(commands.count('home'), int(home))
                self.assertLess(commands.index('hp 1 30'), commands.index('attach 1'))

    def test_servo_fault_stops_startup_without_power_or_home_retry(self):
        from greenhouse_scara_common.robot import TCSMotionClient, TCSCommandError
        commands = []
        def command(text, **kwargs):
            commands.append(text)
            if text == 'state': raise TCSCommandError(text, '-3101')
            return '0'
        with patch('greenhouse_scara_common.robot.telnetlib.Telnet'), \
             patch('greenhouse_scara_common.robot.time.sleep'):
            robot = TCSMotionClient(initialize_motion=False)
            robot.command = command
            with self.assertRaisesRegex(TCSCommandError, 'PID output'):
                robot.enable_motion(home=True)
        self.assertEqual(commands, ['SelectRobot 1', 'state', 'halt'])

    def test_motion_activation_reuses_read_connection(self):
        from greenhouse_scara_common.robot import TCSMotionClient
        with patch('greenhouse_scara_common.robot.telnetlib.Telnet') as telnet, \
             patch.object(TCSMotionClient, '_startup') as startup, \
             patch.object(TCSMotionClient, '_configure_motion') as configure:
            telnet.return_value.read_until.return_value = b'0 100 0 0 0\n'
            robot = TCSMotionClient(initialize_motion=False)
            startup.assert_not_called()
            configure.assert_not_called()
            np.testing.assert_allclose(robot.get_joint_vector(), [0.1, 0, 0, 0])
            robot.enable_motion()
            startup.assert_called_once()
            configure.assert_called_once()
            telnet.assert_called_once()
            telnet.return_value.close.assert_not_called()
            robot.close()


class DownloadTests(unittest.TestCase):
    def make_run(self, root):
        import torch
        complete = root / 'remote/act_rgb_scissor_100'
        complete.mkdir(parents=True)
        torch.save({'weight': torch.arange(4)}, complete / 'policy_last.ckpt')
        for name in ('config.pkl', 'dataset_stats.pkl'):
            (complete / name).write_bytes(pickle.dumps({'original': True}))
        return complete

    def test_completed_runs_only_and_atomic_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = self.make_run(root)
            original = (complete / 'policy_last.ckpt').read_bytes()
            failed = root / 'remote/act_rgb_scissor_101'
            failed.mkdir()
            (failed / 'policy_last.ckpt.tmp').write_bytes(b'partial')
            entries = fetch.discover(root / 'remote', ['act_rgb', 'act_rgbd'])
            self.assertEqual(entries, {})  # Never silently fetch another older job.
            entries = fetch.discover(root / 'remote', ['act_rgb', 'act_rgbd'], job_ids=[100])
            self.assertEqual(set(entries), {'act_rgb'})
            self.assertEqual(entries['act_rgb']['run'], complete.name)
            fetch.download('act_rgb', entries['act_rgb'], root / 'local')
            current = root / 'local/act_rgb/current'
            previous = current.resolve()
            self.assertEqual(pickle.loads((current / 'config.pkl').read_bytes()), {'original': True})
            with patch.object(fetch.shutil, 'copyfile', side_effect=OSError('connection lost')):
                with self.assertRaises(OSError):
                    fetch.download('act_rgb', entries['act_rgb'], root / 'local')
            self.assertEqual(current.resolve(), previous)
            self.assertEqual((current / 'policy_last.ckpt').read_bytes(), original)
            fetch.download('act_rgb', entries['act_rgb'], root / 'local')
            self.assertNotEqual(current.resolve(), previous)
            self.assertTrue(previous.is_dir())

    def test_truncated_checkpoint_never_replaces_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); complete = self.make_run(root)
            entry = fetch.discover(root / 'remote', ['act_rgb'])['act_rgb']
            fetch.download('act_rgb', entry, root / 'local')
            current = root / 'local/act_rgb/current'; previous = current.resolve()
            checkpoint = complete / 'policy_last.ckpt'
            checkpoint.write_bytes(checkpoint.read_bytes()[:100])
            with self.assertRaises(zipfile.BadZipFile):
                fetch.download('act_rgb', entry, root / 'local')
            self.assertEqual(current.resolve(), previous)

    def test_source_change_after_copy_never_replaces_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); complete = self.make_run(root)
            entry = fetch.discover(root / 'remote', ['act_rgb'])['act_rgb']
            fetch.download('act_rgb', entry, root / 'local')
            current = root / 'local/act_rgb/current'; previous = current.resolve()
            copyfile = fetch.shutil.copyfile
            def changing(source, destination):
                copyfile(source, destination)
                if Path(source).name == 'dataset_stats.pkl':
                    (complete / 'config.pkl').write_bytes(pickle.dumps({'changed': True}))
            with patch.object(fetch.shutil, 'copyfile', side_effect=changing):
                with self.assertRaisesRegex(RuntimeError, 'changed during transfer'):
                    fetch.download('act_rgb', entry, root / 'local')
            self.assertEqual(current.resolve(), previous)

    def test_one_ssh_master_is_reused_and_closed_on_failure(self):
        result = SimpleNamespace(returncode=0, stdout='{}', stderr='')
        with patch.object(fetch.subprocess, 'run', return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                with fetch.ssh_connection('user@host') as options:
                    fetch.discover('/remote', ['act_rgb'], 'user@host', options)
                    fetch.run_source(fetch.FINGERPRINT, ['[]'], 'user@host', options)
                    raise RuntimeError('simulated')
            commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(sum('-M' in command for command in commands), 1)
        socket = commands[0][commands[0].index('-S') + 1]
        self.assertTrue(all(command[command.index('-S') + 1] == socket for command in commands))
        self.assertIn('BatchMode=yes', commands[1])
        self.assertIn('exit', commands[-1])


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.current = np.array([.5, 0, 1, 0, -1], dtype=np.float32)
        self.action = np.array([.501, .01, 1.01, .01, 1], dtype=np.float32)
        self.settings = dict(joint_limits=dict(zip(NAMES, [[0, 1], [-2, 2], [0, 6], [-17, 17], [-1.01, 1.01]])),
                             max_delta=[.01, .06, .06, .09, 2.02], max_camera_skew_s=.1,
                             max_observation_age_s=2)
        self.robot_commands, self.scissor_commands = [], []
        self.robot = SimpleNamespace(get_joint_vector=lambda: self.current[:4], movej=self.robot_commands.append)
        self.scissor = SimpleNamespace(read=lambda: self.current[4], set_rotation=self.scissor_commands.append)
        self.camera = SimpleNamespace(read=lambda: {})
        self.predictor = SimpleNamespace(n_obs_steps=2, predict=lambda history: self.action[None])
        self.args = SimpleNamespace(chunks=1, action_count=1, start_action=0, execute=False,
                                    show_camera=False, hz=30)

    def test_readonly_never_commands_either_device(self):
        run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, io.StringIO())
        self.assertEqual(self.robot_commands, [])
        self.assertEqual(self.scissor_commands, [])

    def test_execute_routes_four_joints_and_scissor_radians(self):
        self.args.execute = True
        log = io.StringIO()
        run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, log)
        np.testing.assert_allclose(self.robot_commands[0], self.action[:4])
        self.assertEqual(self.scissor_commands, [1.0])
        self.assertEqual(json.loads(log.getvalue().splitlines()[-1])['event'], 'command')

    def test_invalid_actions_cannot_move_either_device(self):
        self.args.execute = True
        for value in (float('nan'), 20., .8):
            with self.subTest(value=value):
                self.action[0] = value
                with self.assertRaises(ValueError):
                    run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, io.StringIO())
                self.assertEqual(self.robot_commands, [])
                self.assertEqual(self.scissor_commands, [])

    def test_roll_targets_clip_both_bounds_and_preserve_prediction(self):
        self.args.execute = True
        self.current[4] = 0
        for raw, expected in [(1.03752, 1.01), (-1.2, -1.01)]:
            self.action[4] = raw
            log = io.StringIO()
            run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, log)
            self.assertEqual(self.scissor_commands[-1], expected)
            record = json.loads(log.getvalue().splitlines()[0])
            self.assertAlmostEqual(record['prediction'][0][4], raw, places=6)
            self.assertEqual(record['selected'][0][4], expected)

    def test_roll_clipping_does_not_hide_invalid_targets_or_feedback(self):
        self.args.execute = True
        for raw in [float('nan'), float('inf'), -float('inf')]:
            self.action[4] = raw
            with self.assertRaises(ValueError):
                run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, io.StringIO())
        self.action[4] = 0
        self.current[4] = 1.1
        with self.assertRaises(ValueError):
            run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, io.StringIO())
        self.assertEqual(self.robot_commands, [])
        self.assertEqual(self.scissor_commands, [])

    def test_step_limiting_uses_measured_state_for_every_command(self):
        self.args.execute = True
        self.args.limit_action_steps = True
        self.args.action_count = 2
        self.action[1:3] += [.124, .153]
        self.predictor.predict = lambda history: np.stack([self.action, self.action])
        initial = self.current.copy()
        def move(target):
            self.robot_commands.append(target.copy())
            self.current[:4] = target
        self.robot.movej = move
        log = io.StringIO()
        run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, log)
        previous = initial[:4]
        for target in self.robot_commands:
            self.assertTrue(np.all(np.abs(target - previous) <= np.asarray(self.settings['max_delta'][:4])))
            previous = target
        self.assertGreater(self.robot_commands[1][1], self.robot_commands[0][1])
        records = [json.loads(line) for line in log.getvalue().splitlines()]
        commands = [record for record in records if record['event'] == 'command_requested']
        self.assertTrue(all(record['step_limited'] for record in commands))
        np.testing.assert_allclose(commands[0]['policy_target'], self.action)

    def test_step_limiting_still_rejects_invalid_targets(self):
        self.args.execute = True
        self.args.limit_action_steps = True
        for value in (float('nan'), 20.):
            self.action[0] = value
            with self.assertRaises(ValueError):
                run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, io.StringIO())
        self.assertEqual(self.robot_commands, [])
        self.assertEqual(self.scissor_commands, [])

    def test_stale_scissor_prevents_commands(self):
        self.args.execute = True
        def stale(): raise TimeoutError('feedback stale')
        self.scissor.read = stale
        with self.assertRaises(TimeoutError):
            run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, io.StringIO())
        self.assertEqual(self.robot_commands, [])
        self.assertEqual(self.scissor_commands, [])

    def test_scissor_rejection_prevents_robot_command(self):
        self.args.execute = True
        def reject(value): raise RuntimeError('rejected')
        self.scissor.set_rotation = reject
        with self.assertRaises(RuntimeError):
            run_loop(self.predictor, self.camera, self.robot, self.scissor, self.settings, self.args, io.StringIO())
        self.assertEqual(self.robot_commands, [])

    def test_success_key_stops_before_sending_commands(self):
        self.args.execute = True
        self.args.show_camera = True
        with patch('greenhouse_scara_common.deployment.preview_outcome', return_value='success'):
            outcome = run_loop(self.predictor, self.camera, self.robot, self.scissor,
                               self.settings, self.args, io.StringIO())
        self.assertEqual(outcome, 'success')
        self.assertEqual(self.robot_commands, [])
        self.assertEqual(self.scissor_commands, [])

    def test_invalid_configuration_rejected(self):
        for delta in ([1]*4, [1,1,1,1,float('nan')], [1,1,1,1,0]):
            with self.assertRaises(ValueError):
                validate_actions(self.action[None], self.current, self.settings['joint_limits'], delta)

    def test_position_checks_can_be_disabled_but_scissor_and_delta_still_apply(self):
        current = self.current.copy()
        current[0] = 1.137
        target = current.copy()
        target[0] += .001
        validate_actions(target[None], current, self.settings['joint_limits'], self.settings['max_delta'], False)
        with self.assertRaises(ValueError):
            validate_actions(target[None], current, self.settings['joint_limits'], self.settings['max_delta'], True)
        target[4] = 1.2
        with self.assertRaises(ValueError):
            validate_actions(target[None], current, self.settings['joint_limits'], self.settings['max_delta'], False)


if __name__ == '__main__':
    unittest.main()
