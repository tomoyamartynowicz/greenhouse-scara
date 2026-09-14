# DP3 voor SCARA

Officiële DP3 PointNet en conditional U-Net, aangepast aan vier joints en vier
absolute next-qpos-acties. De modelcode staat met MIT-licentie in
`../greenhouse_scara_3d_common/vendor/dp3`; commit
`47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97` van
[YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy).

De DP3-policy, PointNet en U-Net zijn met die commit vergeleken: gelijk na
namespace-aanpassing en verwijdering van één ongebruikte PyTorch3D-import.
Zie [audit](../../docs/dp3_source_audit.json) en
[implementatiekeuzes](../../docs/DP3_IMPLEMENTATION.md).

## Pointcloud bekijken

Open [inspect_pointcloud.ipynb](../../inspect_pointcloud.ipynb) met de lokale
`act`-kernel. Of voer vanuit de repositoryroot uit:

```bash
python scripts/inspect_pointcloud.py
```

Het resultaat staat onder `runs/pointcloud_check`: interactieve standalone HTML,
RGB/depth-preview, volledige XYZRGB-PLY, de 1.024 modelpunten als PLY/NPY en een
JSON-diagnose. Voor frame 0 zijn de bestanden al lokaal gemaakt.

## Invoer

De meegeleverde calibratie hoort bij jouw `camera_top`, D405 serial
`130322273198`, RGB-aligned depth op 640×480. De `Color`-intrinsics en vervorming
komen uit `calibration/realsense_calibration-130322273198.txt`. Dat bestand
beschrijft interne sensorextrinsics, geen camera→robot- of bottom→top-pose.
De HDF5 bevat geen cameraserial; de koppeling volgt de aangeleverde bestanden.
Bij twee camera's is een gemeten onderlinge pose nodig voordat je fuseert.

Standaard `config.yaml`:

- Geen XYZ-crop (`crop_min/max: null`), geen bovengrens op depth.
- Alle niet-nul depthpixels op de originele resolutie (`pixel_stride: 1`).
- FPS op **alle** geldige punten (`fps_candidates: null`), naar 1.024 XYZ-punten.
- XYZ-only, zoals DP3; `use_color: true` geeft XYZRGB, geen RGB-imagebackbone.
- Per-as limits-normalisatie, gefit op alleen trainingsepisodes.
- Twee observaties, horizon 16, acht uitvoeracties, tien DDIM-stappen.
- Encoder-output 64, U-Net `[512,1024,2048]`, `prediction_type: sample`.

Volledige observatie betekent hier geen ruimtelijk weggesneden werkgebied;
1.024 punten representeren wel een reductie. Zonder crop kunnen verre achtergrond
of slechte dieptemetingen een deel van het puntenbudget innemen. FPS op alle
punten kost CPU-tijd; `fps_candidates: 4096` is een snellere expliciete benadering.
De viewer markeert hoeveel waarden op de maximumcode zitten en verwijdert ze niet.

## Lokale technische training

Vanuit de repositoryroot, in de werkende diffusion/act-omgeving:

```bash
python -m pip install -r environments/requirements-dp3.txt
python src/greenhouse_scara_3d_diffusion_policy/train.py \
  allow_single_episode=true device=cpu epochs=1 max_train_steps=1 max_val_steps=1 \
  batch_size=1 val_batch_size=1 use_ema=false warmup_steps=0 \
  'policy.down_dims=[32,64]' policy.diffusion_step_embed_dim=32 \
  policy.num_inference_steps=1 output_dir=runs/dp3_cpu_test
```

De standaarddataset is de echte lokale `../datasets/test_rgbd_single_joints`.
Dit commando verkleint alleen het netwerk voor een CPU-test. De gewone config
en de Slurm-smoketest behouden de volledige DP3-architectuur. Met één demo is
validatie dezelfde opname: uitsluitend een technische controle.

## DelftBlue

Kopieer de echte demo-map naar scratch; gebruik hiervoor niet de 2D-dummy-depth
met de echte camera-calibratie. Calibratie en modelcode staan samen in Git.
De diffusion-Pixi-omgeving heeft aanvullend `termcolor` en `pyrealsense2` nodig.
Plotly is alleen voor de viewer. Er is geen aangesloten camera nodig voor
SDK-deprojectie en er is geen PyTorch3D/simulator nodig voor deze route.

```bash
cd ~/scara_ws/greenhouse-scara
export GREENHOUSE_PIXI_PROJECT=/scratch/$USER/thesis/envs/diffusion_policy
export GREENHOUSE_DP3_DATASET_DIR=/scratch/$USER/thesis/datasets/test_rgbd_single_joints
bash scripts/submit.sh dp3-smoke
```

De datasetvariabele is bewust `GREENHOUSE_DP3_DATASET_DIR`; de 2D-instelling
`GREENHOUSE_DATASET_DIR` kan nog naar dummydata wijzen. Installeer de aanvullende
packages via Pixi in de bestaande omgeving voordat je de job indient.

`dp3-smoke` gebruikt één train- en validatiestap, batch 1, geen EMA en de echte
volledige PointNet/U-Net. Checkpoints staan onder
`/scratch/$USER/thesis/runs/greenhouse_scara_3d_diffusion_policy/job<JOBID>/latest.ckpt`
en `best.ckpt`; beide bewaren ook calibratie en normalisatie.
Voor normale training gebruik je `dp3` en minimaal twee episodes.
De GPU-run op DelftBlue is nog niet uitgevoerd tijdens deze implementatie.

Relatieve configpaden zijn vanaf de repositoryroot. Override `dataset_dir=...`,
`calibration_file=...` en `output_dir=...` met absolute paden waar nodig.
De oude lokale `scara_ws/src/greenhouse_scara_3d_*`-paden blijven via symlinks werken.
