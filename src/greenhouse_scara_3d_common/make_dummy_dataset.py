"""Copy real RGB/joints into a separate dataset with clearly synthetic depth."""
from pathlib import Path
import json
import copy
import numpy as np
import h5py
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = WORKSPACE / 'datasets/leaf_cutting_experiment_rgb'
DESTINATION = WORKSPACE / 'datasets/leaf_cutting_dummy_rgbd_20'
NUM_DEMOS = 20
SOURCE_CAMERA = 'wrist_d405'
CAMERA = 'camera_top'
MODEL_FOLDERS = {
    'dp3': 'greenhouse_scara_3d_diffusion_policy',
    'flow3d': 'greenhouse_scara_3d_flow_matching_policy',
    'act3d': 'greenhouse_scara_3d_act',
    'maniflow': 'greenhouse_scara_maniflow',
}


def dummy_depth(height, width, frame, frames, episode):
    """Artificial sloping surface and moving bump, in millimetres; no RGB inference."""
    y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    phase = 2 * np.pi * frame / max(1, frames - 1) + episode * .2
    center_x, center_y = .45 * np.sin(phase), .3 * np.cos(phase)
    bump = np.exp(-((x-center_x)**2 + (y-center_y)**2) / .12)
    millimetres = np.rint(340 + 30*x + 20*y - 65*bump + 10*np.sin(phase)).astype(np.uint16)
    millimetres[:4] = 0  # A small invalid border exercises the depth validity mask.
    return millimetres


def write_configs(destination):
    folder = destination / 'configs'
    folder.mkdir()
    for model, project in MODEL_FOLDERS.items():
        cfg = yaml.safe_load((WORKSPACE / 'src' / project / 'config.yaml').read_text())
        cfg.update(dataset_dir=str(destination), calibration_file=str(destination / 'calibration.yaml'),
                   output_dir=str(WORKSPACE / 'runs/scara_3d_dummy' / model / 'full'))
        (folder / (model + '.yaml')).write_text(yaml.safe_dump(cfg, sort_keys=False))
        smoke = copy.deepcopy(cfg)
        smoke.update(device='cpu', batch_size=4, val_batch_size=4, epochs=1, num_workers=0,
                     warmup_steps=0, max_train_steps=5, max_val_steps=1,
                     output_dir=str(WORKSPACE / 'runs/scara_3d_dummy' / model / 'smoke'))
        smoke['pointcloud'].update(num_points=128, fps_candidates=512, pixel_stride=4, use_color=True)
        policy = smoke['policy']
        if model in ('dp3', 'flow3d'):
            policy.update(down_dims=[32, 64, 128], diffusion_step_embed_dim=32, n_groups=8,
                          num_inference_steps=2)
        elif model == 'act3d':
            policy.update(hidden_dim=64, nheads=4, enc_layers=1, dec_layers=1, dim_feedforward=128)
        else:
            policy.update(n_layer=1, n_head=4, n_emb=64, num_inference_steps=2)
        (folder / (model + '_smoke.yaml')).write_text(yaml.safe_dump(smoke, sort_keys=False))


def main():
    paths = sorted(SOURCE.glob('episode_*.hdf5'), key=lambda p: int(p.stem.split('_')[-1]))[:NUM_DEMOS]
    if len(paths) != NUM_DEMOS:
        raise ValueError(f'Need {NUM_DEMOS} source demos, found {len(paths)}')
    # New folder only: never overwrite source recordings or an earlier generated dataset.
    DESTINATION.mkdir(exist_ok=False)
    manifest = {'synthetic_depth': True, 'synthetic_calibration': True, 'source_dir': str(SOURCE),
                'source_camera': SOURCE_CAMERA, 'camera': CAMERA, 'episodes': [],
                'actions': 'No stored action; loaders derive next_qpos with action_offset=1.',
                'depth': 'Artificial sloping surface plus moving bump; NOT reconstructed scene geometry.'}
    size = None
    for index, source in enumerate(paths):
        target = DESTINATION / f'episode_{index}.hdf5'
        temporary = target.with_suffix('.hdf5.tmp')
        with h5py.File(source, 'r') as original, h5py.File(temporary, 'w') as output:
            images = original[f'observations/images/{SOURCE_CAMERA}']
            joints = original['observations/qpos']
            frames, height, width, channels = images.shape
            if channels != 3 or images.dtype != np.uint8 or joints.shape != (frames, 4):
                raise ValueError(f'{source}: expected RGB (T,H,W,3) and joints (T,4)')
            if not np.isfinite(joints[:]).all(): raise ValueError(f'{source}: nonfinite joints')
            if size is not None and size != (height, width): raise ValueError('Mixed image resolutions')
            size = height, width
            output.attrs.update(synthetic_depth=True, synthetic_calibration=True, source_episode=str(source))
            output.create_dataset('observations/qpos', data=joints[:])
            rgb = output.create_dataset(f'observations/images/{CAMERA}', shape=images.shape, dtype=np.uint8,
                                        chunks=(1,height,width,3), compression='lzf')
            depth = output.create_dataset(f'observations/depth/{CAMERA}', shape=(frames,height,width),
                                          dtype=np.uint16, chunks=(1,height,width), compression='gzip',
                                          compression_opts=1, shuffle=True)
            depth.attrs.update(depth_scale=.001, aligned_to_rgb=True, synthetic=True)
            for i in range(frames):
                rgb[i] = images[i]
                depth[i] = dummy_depth(height, width, i, frames, index)
            for old, new in [('image_time', f'images/{CAMERA}'), ('qpos_time', 'qpos')]:
                times = original[f'timestamps/{old}'][:]
                if times.shape != (frames,) or not np.isfinite(times).all():
                    raise ValueError(f'{source}: invalid {old}')
                output.create_dataset('timestamps/' + new, data=times)
                if old == 'image_time':
                    ds = output.create_dataset(f'timestamps/depth/{CAMERA}', data=times)
                    ds.attrs['synthetic'] = True
        temporary.replace(target)
        manifest['episodes'].append(dict(file=target.name, source=str(source), frames=frames))
        print(f'{index+1}/{NUM_DEMOS}: {target.name}, {frames} frames', flush=True)
    height, width = size
    cal = {'synthetic': True, 'purpose': 'Dummy dataset only; NEVER use for real recordings or live cameras.',
           'cameras': {CAMERA: dict(serial='SYNTHETIC-NOT-A-REAL-CAMERA', width=width, height=height,
              aligned_to_rgb=True, to_reference=None,
              intrinsics=dict(fx=float(width), fy=float(width), cx=(width-1)/2, cy=(height-1)/2,
                              distortion_model='none', coeffs=[0.] * 5))}}
    (DESTINATION / 'calibration.yaml').write_text(yaml.safe_dump(cal, sort_keys=False))
    (DESTINATION / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_configs(DESTINATION)
    count = sum(e['frames'] for e in manifest['episodes'])
    commands = '\n'.join(f'python src/{project}/train.py --config datasets/{DESTINATION.name}/configs/{model}_smoke.yaml'
                         for model, project in MODEL_FOLDERS.items())
    readme = f'''# Dummy RGB-D dataset — alleen technische trainingstests

{NUM_DEMOS} volledige demonstraties, {count} frames, RGB/depth {width}×{height}, camera `{CAMERA}`.
RGB en vier jointposities zijn ongewijzigd gekopieerd uit de eerste {NUM_DEMOS} numeriek gesorteerde
episodes in `{SOURCE}`. `manifest.json` koppelt iedere demo aan de bron.

Depth is **synthetisch**: een schuin oppervlak met een bewegende bult en enkele ongeldige pixels.
Het is geen schatting van de echte scène. `calibration.yaml` is eveneens volledig synthetisch;
gebruik deze kalibratie uitsluitend met deze dummy-dataset.

RGB is verliesloos gecomprimeerd; depth is raw uint16, met `depth_scale=0.001` meter per eenheid.
`aligned_to_rgb=true` betekent hier alleen dat synthetische depth op dezelfde pixelgrid is gemaakt.
Beeld- en jointtimestamps zijn gekopieerd; dummy-depth krijgt de beeldtimestamp.
Dit test geen echte RGB/depth-synchronisatie of ruimtelijke nauwkeurigheid.

Er is geen opgeslagen `action`: de greenhouse-loaders maken volgende jointposities als acties aan.
De bronbestanden, inclusief hun bestaande acties, blijven ongewijzigd.

## Korte CPU-proef

Vanuit `/home/tomoya/scara_ws`, in de `act`-omgeving:

```bash
python -m pip install -r src/greenhouse_scara_3d_common/requirements.txt
export OMP_NUM_THREADS=1
{commands}
```

Deze vier smoke-configuraties gebruiken kleinere modellen, 128 XYZRGB-punten, batchgrootte 4,
vijf trainingsbatches en één validatiebatch. De splitsing is per episode (18 train / 2 validatie).
Checkpoints en metrics komen in `runs/scara_3d_dummy/<model>/smoke`.
Als die run al bestaat: kies een nieuwe `output_dir=...` of hervat met `resume=true epochs=2`.

## Oorspronkelijke modelgroottes testen

Gebruik dezelfde commando's met `<model>.yaml` in plaats van `<model>_smoke.yaml`.
Deze vier configuraties behouden de standaard modelarchitectuur en gebruiken `cuda:0`.
Voeg voor een korte proef `epochs=1 max_train_steps=5 max_val_steps=1` toe.
Deze groottes zijn niet hetzelfde als de kleine CPU-testmodellen; een geslaagde CPU-proef
bevestigt niet dat de standaardmodellen in jouw GPU-geheugen passen.

Een dalende loss op deze dataset zegt niets over de kwaliteit van een echte 3D-policy.
De datasetgenerator staat in `src/greenhouse_scara_3d_common/make_dummy_dataset.py`.
'''
    (DESTINATION / 'README.md').write_text(readme)
    print(f'Created {DESTINATION}: {NUM_DEMOS} demos, {count} frames', flush=True)


if __name__ == '__main__': main()
