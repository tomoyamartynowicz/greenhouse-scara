#!/usr/bin/env python3
"""Download DelftBlue checkpoints over SSH/rsync into the notebook's local folders."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def fetch(kind, remote, host, output, dry_run=False):
    path = PurePosixPath(remote)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Gebruik een absoluut checkpointpad op DelftBlue, zonder ..')
    files = [path]
    if kind == 'act':
        files += [path.parent / 'config.pkl', path.parent / 'dataset_stats.pkl']
    destination = output / kind
    sources = [f'{host}:{p}' for p in files]
    command = ['rsync', '--protect-args', '--times', '--progress', '--']
    if dry_run:
        print(shlex.join(command + sources + [str(destination) + '/']))
        return
    output.mkdir(parents=True, exist_ok=True)
    # Publish locally only after all required files have arrived successfully.
    with tempfile.TemporaryDirectory(prefix='.download-', dir=output) as tmp:
        subprocess.run(command + sources + [tmp + '/'], check=True)
        for item in files:
            if not (Path(tmp) / item.name).is_file():
                raise FileNotFoundError(f'Niet ontvangen: {item}')
        destination.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(files):
            name = ('policy_last.ckpt' if kind == 'act' else 'latest.ckpt') if index == 0 else item.name
            os.replace(Path(tmp) / item.name, destination / name)
        (destination / 'download_source.json').write_text(
            json.dumps({'host': host, 'files': [str(p) for p in files]}, indent=2) + '\n')
    print(f'{kind}: opgeslagen in {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='tomoyamartynow@login.delftblue.tudelft.nl')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without connecting')
    for kind in ('act', 'dp', 'fm', 'dp3', 'fm3'):
        parser.add_argument(f'--{kind}', help='Absoluut pad naar het remote checkpointbestand')
    args = parser.parse_args()
    if args.host.startswith('-') or any(c.isspace() for c in args.host) or ':' in args.host:
        parser.error('Gebruik een SSH-hostnaam of user@host')
    selected = [(k, getattr(args, k)) for k in ('act', 'dp', 'fm', 'dp3', 'fm3') if getattr(args, k)]
    if not selected:
        parser.error('Geef minimaal één checkpoint op: --act, --dp, --fm, --dp3 of --fm3')
    if not args.dry_run and shutil.which('rsync') is None:
        parser.error('rsync is niet geïnstalleerd op deze laptop')
    for kind, remote in selected:
        fetch(kind, remote, args.host, args.output.expanduser().resolve(), args.dry_run)


if __name__ == '__main__':
    main()
