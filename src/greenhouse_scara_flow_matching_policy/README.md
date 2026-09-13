# Greenhouse SCARA Flow Matching

Aparte kopie met eigen Python-package `greenhouse_flow_matching_policy`, meerdere RGB/depth-streams en
actieafleiding tijdens laden. Zie ook [repository- en clusterinstructies](../../README.md). Zie [gedeelde uitleg](../greenhouse_scara_common/README.md)
voor het inputformaat, chunks, depth en tests. De oorspronkelijke broncode en upstream-documentatie staan in de afzonderlijke
`scara_flow_matching_policy`-checkout; die is geen trainingsafhankelijkheid.

## Inputs

Pas `greenhouse_flow_matching_policy/config/task/scara_image.yaml` aan. In `shape_meta.obs` staan
standaard RGB en depth van `camera_top`, plus joints. Voeg entries toe voor
`camera_bottom`, of verwijder de depth-entry voor alleen RGB. Dataset, model en
evaluator verwijzen naar dezelfde `shape_meta`; command-line-overrides blijven
zo ook overal gelijk.

`action_source: auto` gebruikt bestaande acties, of leidt ze af uit joints.
`action_offset: 1` betekent de volgende jointpositie, onafhankelijk van `horizon`.

## Trainen

Gebruik de bestaande modelomgeving, bijvoorbeeld de lokale `act`-omgeving:

```bash
cd ~/greenhouse-scara/src/greenhouse_scara_flow_matching_policy
export WANDB_MODE=offline

python train.py --config-name=train_flow_matching_unet_scara_workspace \
  task.dataset_path=/pad/naar/greenhouse_demos \
  run_name=greenhouse_rgbd_run01
```

Defaults: 20 epochs, batch 64, 2 observaties, horizon 16, 8 uitvoeracties, CUDA,
EMA en een eigen ResNet18 per stream. RGB/depth hebben aparte normalisatie en
geen willekeurige crops. De visuele gewichten beginnen standaard zonder pretraining.
Verklein batchgrootte bij extra streams als het GPU-geheugen beperkt is.

Runs staan standaard in `$HOME/greenhouse_runs/flow_matching_policy/<run_name>`.
`GREENHOUSE_FM_RUN_ROOT` overschrijft de root. `GREENHOUSE_DATASET_DIR` kan het datasetpad leveren;
een expliciete `task.dataset_path=...` heeft voorrang. Standaard `training.resume=false`.
Hervatten met `training.resume=true` zoekt naar `checkpoints/latest.ckpt` in dezelfde
runmap; gebruik daarbij dezelfde inputconfiguratie. De geërfde loop voert bij
hervatten opnieuw `training.num_epochs` epochs uit.

Kleine CPU-controle met de ene voorbeeldopname:

```bash
OMP_NUM_THREADS=1 python train.py --config-name=train_flow_matching_unet_scara_workspace \
  task.dataset_path=/home/tomoya/scara_ws/datasets/test_rgbd_single_joints \
  task.allow_single_episode=true run_name=greenhouse_rgbd_smoke01 \
  training.device=cpu training.num_epochs=1 training.use_ema=false \
  training.max_train_steps=1 training.max_val_steps=1 logging.mode=disabled \
  dataloader.batch_size=1 dataloader.num_workers=0 dataloader.persistent_workers=false \
  val_dataloader.batch_size=1 val_dataloader.num_workers=0 val_dataloader.persistent_workers=false \
  task.env_runner.n_test_samples=1 task.env_runner.batch_size=1 \
  policy.num_inference_steps=1 'policy.down_dims=[16,32]' \
  policy.diffusion_step_embed_dim=16 policy.n_groups=4
```

Dit gebruikt een kleinere U-Net en dezelfde demo voor training/validatie. Het test
de techniek, niet de generalisatie. De voorbeeld-HDF5 hoeft geen `action` te hebben.
Een voltooide run bewaart `checkpoints/latest.ckpt`, `logs.json.txt` en `.hydra/config.yaml`.

Voor de cluster is `train_flow_matching.slurm` aangepast aan de nieuwe broncode en aparte runmap.
Controleer daarin het scratch-datasetpad en de bestaande Pixi-omgeving vóór `sbatch`.

## Live-inference

```bash
python eval_scara.py --checkpoint /pad/naar/checkpoints/latest.ckpt --chunks 1
```

Dit leest de geconfigureerde RGB/depth-streams uit het checkpoint en start standaard
zonder robotbeweging. Gebruik `--camera NAME SERIAL` voor camera's met andere
serienummers. Bestaande joint-limit- en maximale-deltacontroles zijn behouden;
`--execute` schakelt beweging in. De RGB-preview ontvangt niet de genormaliseerde
modelinput; de policy krijgt alle geselecteerde streams met trainingspreprocessing.

De flow-loss en Euler/Heun-solver zijn behouden; standaard 16 Heun-stappen.
Diffusioncheckpoints zijn niet uitwisselbaar met flowcheckpoints.

## Opgeruimde projectstructuur

Alleen SCARA-taskconfiguraties, datasets, runners en hun trainingsworkspaces zijn
behouden. PushT, Kitchen, Block Pushing, simulatie-assets, andere robotdemo's en
de bijbehorende tests/configuraties zijn verwijderd, samen met oude upstream
Conda-recepten en voorbeeldmedia. Gebruik de bestaande modelomgeving zoals hierboven.

`model/` en `policy/` zijn volledig behouden, inclusief hun geïmporteerde helpers;
er zijn geen architecturen, losses, normalisatoren of samplers verwijderd of gewijzigd.
Daarom staan er nog generieke modelvarianten en `robomimic_config_util.py`:
die horen bij de behouden modelcode, niet bij een actieve simulatietaak.
Ook checkpointcode, offline/live-evaluatie, Slurm en de optionele Ray-tools blijven aanwezig.

## RGB-test op 480×640

De repository-wrapper `scripts/submit.sh fm-smoke` selecteert
`train_flow_matching_unet_scara_smoke.yaml` met `task/scara_rgb_dual.yaml`.
Net als `dp-smoke`: twee RGB-camera's op 480×640 zonder crops, qpos,
twee observatiestappen, horizon 16, acht uitvoeracties en de volledige huidige
U-Net `[256,512,1024]`. Er wordt één train- en validatiestap uitgevoerd met
batch 1 en zonder EMA; `checkpoints/latest.ckpt` wordt opgeslagen.
De flow-loss en 16 Heun-stappen blijven behouden. De normale RGB-D-taak is niet
gewijzigd. Zie de [repository-instructies](../../README.md) voor clusterpaden.
