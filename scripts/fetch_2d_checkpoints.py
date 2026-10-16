#!/usr/bin/env python3
"""Fetch complete 2D checkpoints from DelftBlue; never write to the cluster."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import pickletools
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = tuple(f'{model}_{mode}' for model in ('act', 'dp', 'fm') for mode in ('rgb', 'rgbd'))
# Run this stdlib-only discovery code locally on a mounted SFTP tree, or over SSH.
DISCOVERY = r'''
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
variants = json.loads(sys.argv[2])
job_ids = set(json.loads(sys.argv[3]))
result = {}
for variant in variants:
    candidates = []
    for run in root.glob(variant + '_scissor_*'):
        if not re.fullmatch(re.escape(variant) + r'_scissor_\d+', run.name):
            continue
        job_id = int(run.name.rsplit('_', 1)[1])
        if job_ids and job_id not in job_ids:
            continue
        names = ['policy_last.ckpt', 'config.pkl', 'dataset_stats.pkl'] if variant.startswith('act_') else ['checkpoints/latest.ckpt']
        files = [run / name for name in names]
        candidates.append((job_id, run, files))
    if candidates:
        _, run, files = max(candidates, key=lambda item: item[0])
        if all(p.is_file() and p.stat().st_size > 0 for p in files):
            result[variant] = dict(run=run.name, files=[str(p) for p in files])
print(json.dumps(result))
'''

FINGERPRINT = r'''
import hashlib, json, pathlib, sys
result = {}
for name in json.loads(sys.argv[1]):
    path = pathlib.Path(name)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('Source changed while hashing: ' + name)
    result[name] = dict(size=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest.hexdigest())
print(json.dumps(result))
'''


@contextmanager
def ssh_connection(host):
    """Authenticate once; all subsequent operations reuse this private socket."""
    if host is None:
        yield []
        return
    with tempfile.TemporaryDirectory(prefix='scara-ssh-', dir='/tmp') as directory:
        socket = str(Path(directory) / 'control')
        try:
            print(f'Connecting once to {host} for all selected downloads...', flush=True)
            subprocess.run(['ssh', '-M', '-S', socket, '-o', 'ControlMaster=yes',
                            '-o', 'ControlPersist=600', '-o', 'ServerAliveInterval=30',
                            '-o', 'ServerAliveCountMax=3', '-fN', host], check=True)
            # A lost connection must fail instead of asking for six more passwords.
            yield ['-S', socket, '-o', 'ControlMaster=no', '-o', 'BatchMode=yes']
        finally:
            subprocess.run(['ssh', '-S', socket, '-O', 'exit', host],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_source(script, arguments, host=None, ssh_options=()):
    if host:
        command = ['ssh', *ssh_options, host, shlex.join(['python3', '-', *arguments])]
    else:
        import sys
        command = [sys.executable, '-', *arguments]
    result = subprocess.run(command, input=script, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'Reading checkpoint source failed')
    return json.loads(result.stdout)


def discover(source_root, variants, host=None, ssh_options=(), job_ids=()):
    arguments = [str(source_root), json.dumps(list(variants)), json.dumps(list(job_ids))]
    return run_source(DISCOVERY, arguments, host, ssh_options)


def validate_serialization(path):
    if path.suffix == '.ckpt':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not any(name.endswith('/data.pkl') for name in names) or not any('/data/' in name for name in names):
                raise ValueError(f'{path.name}: missing PyTorch checkpoint payload/tensors')
            damaged = archive.testzip()
            if damaged is not None:
                raise ValueError(f'{path.name}: damaged checkpoint member {damaged}')
    else:
        operations = list(pickletools.genops(path.read_bytes()))
        if not operations or operations[-1][0].name != 'STOP' or operations[-1][2] != path.stat().st_size - 1:
            raise ValueError(f'{path.name}: incomplete pickle file')


def download(variant, entry, output, host=None, ssh_options=()):
    directory = output / variant
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=directory) as temporary:
        stage = Path(temporary)
        files = entry['files']
        print(f"{variant}: {entry['run']} — checking source and downloading", flush=True)
        before = run_source(FINGERPRINT, [json.dumps(files)], host, ssh_options)
        if host:
            command = ['rsync', '-e', shlex.join(['ssh', *ssh_options]), '--protect-args', '--times', '--progress', '--']
            subprocess.run(command + [f'{host}:{p}' for p in files] + [str(stage) + '/'], check=True)
        else:
            for path in files:
                shutil.copyfile(path, stage / Path(path).name)
        hashes = {}
        for path in files:
            local = stage / Path(path).name
            if not local.is_file() or not local.stat().st_size:
                raise RuntimeError(f'Missing/empty download: {local.name}')
            digest = hashlib.sha256()
            with local.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            hashes[local.name] = digest.hexdigest()
            if local.stat().st_size != before[path]['size'] or hashes[local.name] != before[path]['sha256']:
                raise RuntimeError(f'{local.name}: source/download mismatch; stop the job and retry')
            validate_serialization(local)
        after = run_source(FINGERPRINT, [json.dumps(files)], host, ssh_options)
        if before != after:
            raise RuntimeError(f"{entry['run']}: checkpoint changed during transfer; stop the job and retry")
        metadata = dict(variant=variant, host=host, run=entry['run'], files=files,
                        downloaded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), sha256=hashes,
                        source_verified=after, serialization_checked=True)
        (stage / 'download_source.json').write_text(json.dumps(metadata, indent=2) + '\n')
        destination = directory / (entry['run'] + '_' + str(time.time_ns()))
        # Publish the entire bundle together; a failed download leaves current intact.
        stage.rename(destination)
        link = directory / ('.current-' + str(time.time_ns()))
        link.symlink_to(destination.name, target_is_directory=True)
        os.replace(link, directory / 'current')
    print(f'{variant}: {directory / "current"} <- {entry["run"]}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='tomoyamartynow@login.delftblue.tudelft.nl')
    parser.add_argument('--remote-root', default='/scratch/tomoyamartynow/thesis/runs/2d')
    parser.add_argument('--source-root', type=Path, help='Read a locally mounted DelftBlue runs/2d folder instead of SSH')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints/2d')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--job-ids', nargs='+', type=int, help='Only consider these training job IDs (recommended after cancelling selected jobs)')
    parser.add_argument('--list', action='store_true', help='Only list complete available checkpoints')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.host) or args.host.startswith('-'):
        parser.error('Invalid SSH host; use user@hostname')
    if not Path(args.remote_root).is_absolute() or '..' in Path(args.remote_root).parts:
        parser.error('--remote-root must be an absolute path without ..')
    host = None if args.source_root else args.host
    if host and not args.list and not shutil.which('rsync'):
        parser.error('rsync is required for SSH downloads')
    if args.job_ids and any(job < 1 for job in args.job_ids):
        parser.error('--job-ids must be positive')
    with ssh_connection(host) as ssh_options:
        entries = discover(args.source_root or args.remote_root, args.variants, host, ssh_options, args.job_ids or ())
        missing = [v for v in args.variants if v not in entries]
        for variant in missing:
            print(f'{variant}: latest selected run has no complete checkpoint bundle (or no matching run); not falling back to older jobs')
        if args.list:
            print(json.dumps(entries, indent=2))
        else:
            for variant, entry in entries.items():
                download(variant, entry, args.output.expanduser().resolve(), host, ssh_options)
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
