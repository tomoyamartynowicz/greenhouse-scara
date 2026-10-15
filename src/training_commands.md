# Trainingscommando's

Voor de zes 2D-DelftBlue-jobs met episodefilter vanaf 11 en aparte RGB/RGBD-
checkpoints: zie [docs/2D_DELFTBLUE.md](../docs/2D_DELFTBLUE.md).

Voer eerst de voorbereiding uit, daarna het gewenste modelcommando.
De datasetmap bevat episodes vanaf 11 als symlinks naar de oorspronkelijke bestanden.
Start bij een nieuwe run met een nieuwe `RUN`-waarde. Epochs en batchgroottes hieronder
zijn voorbeelden, geen afgestemde hyperparameters.

```bash
conda activate act
cd /home/tomoya/scara_ws/greenhouse-scara
DATASET=/home/tomoya/scara_ws/datasets/green_house_dataset1_from11
RUN=$(date +%Y%m%d_%H%M%S)
```

## ACT — RGB, inclusief schaar-roll

Twee RGB-camera's op 480×640, vijf observatie- en actiedimensies.

```bash
python src/greenhouse_scara_act/imitate_episodes.py \
  --dataset_dir "$DATASET" \
  --inputs_config src/greenhouse_scara_act/inputs_rgb_dual_scissor.yaml \
  --ckpt_dir "runs/act_rgb_$RUN" \
  --device cuda --batch_size 8 --num_workers 4 \
  --num_epochs 200 --chunk_size 30 --lr 1e-5
```

## DP 2D — RGB, inclusief schaar-roll

Twee RGB-camera's op 480×640. De volgorde is `[J1, J2, J3, J4, scissor_roll]`.
De observatie gebruikt de gemeten `observations/scissor_rotation`; de vijfde actie
gebruikt `actions/scissor_rotation` van hetzelfde tijdstip. Alleen de vier
robotacties worden bij ontbrekende `/action` afgeleid uit de volgende gemeten qpos.
Er worden geen HDF5-bestanden aangepast.

### Korte trainingstest op CPU

Dit gebruikt de volledige huidige U-Net `[256,512,1024]` en beide RGB-encoders.
Per epoch: één trainbatch en één validatiebatch, batchgrootte 1. Daarnaast draait
offline actievoorspelling op één validatiesample. De vijfde dimensie wordt ook
genormaliseerd en geëvalueerd. Checkpoint: `checkpoints/latest.ckpt` in de runmap.

```bash
OMP_NUM_THREADS=4 python src/greenhouse_scara_diffusion_policy/train.py \
  --config-name=train_diffusion_unet_scara_smoke \
  task=scara_rgb_dual_scissor task.dataset_path="$DATASET" \
  output_dir="$PWD/runs/dp_rgb_scissor_smoke_$RUN" run_name="dp_rgb_scissor_smoke_$RUN" \
  training.device=cpu training.num_epochs=2 training.lr_warmup_steps=0 \
  policy.num_inference_steps=10
```

De smoke-config schakelt EMA en W&B uit en zet dataloader-workers op 0.
Warmup staat hier uit zodat beide optimizerstappen een niet-nul learning rate hebben.
Tien DDIM-stappen versnellen alleen de actievoorspelling tijdens deze test;
de diffusion-training behoudt 100 noise-timesteps en dezelfde loss.
Dit controleert de trainingstechniek, niet de kwaliteit van het geleerde gedrag.

Getest op 13 oktober 2026 op de echte dataset: 171 trainingsepisodes en 19
validatie-episodes, twee optimizerstappen en twee validatiebatches geslaagd.
Ook de gewichten van de vijfde uitvoerdimensie zijn bijgewerkt. Het checkpoint
is opnieuw geladen en produceert eindige actiechunks met shape `[1, 16, 5]`.
Run: `runs/dp_rgb_scissor_smoke_20261013_QhHp54` (inclusief `train.log`,
`logs.json.txt` en `verification.json`).

### Volledige training op GPU

De gewone configuratie behoudt EMA, learning-rate-warmup en 100 DDIM-stappen.
Bij onvoldoende GPU-geheugen verlaag je beide batchgroottes.

```bash
python src/greenhouse_scara_diffusion_policy/train.py \
  --config-name=train_diffusion_unet_scara_workspace \
  task=scara_rgb_dual_scissor task.dataset_path="$DATASET" \
  output_dir="$PWD/runs/dp_rgb_scissor_$RUN" run_name="dp_rgb_scissor_$RUN" \
  training.device=cuda:0 training.num_epochs=200 \
  dataloader.batch_size=8 val_dataloader.batch_size=8 logging.mode=offline
```

De roll-uitbreiding verandert uitsluitend de datasetadapter, input/outputdimensies
en doorgifte aan de offline evaluator. De bestaande U-Net, diffusion-loss,
DDIM-scheduler, sequence-padding en normalisatiemethode blijven behouden.
De greenhouse-versie had al een eigen RGB-encoder/invoeradapter; deze wijziging
maakt de gehele implementatie dus niet identiek aan de originele DP-repository.
Oude vierdimensionale runs blijven werken met `task=scara_rgb_dual`.
Vierdimensionale checkpoints kunnen niet hervat worden als vijfdimensionale runs.
De live robotrunner ondersteunt de schaar-roll nog niet.

## Flow Matching 2D — RGB, inclusief schaar-roll

Twee RGB-camera's, vier robotjoints en schaar-roll.

```bash
python src/greenhouse_scara_flow_matching_policy/train.py \
  --config-name=train_flow_matching_unet_scara_workspace \
  task=scara_rgb_dual_scissor task.dataset_path="$DATASET" \
  output_dir="$PWD/runs/fm_rgb_$RUN" run_name="fm_rgb_$RUN" \
  training.device=cuda:0 training.num_epochs=200 \
  dataloader.batch_size=8 val_dataloader.batch_size=8 logging.mode=offline
```

## DP3 — pointcloud, nog zonder schaar-roll

Alleen de bovenste camera, 1024 XYZ-punten zonder kleur, vier joints.
De calibratie hoort bij RGB-aligned depth van de bovenste D405.
De eerste normalisatiescan en pointcloud-sampling kunnen lang duren.

```bash
python src/greenhouse_scara_3d_diffusion_policy/train.py \
  dataset_dir="$DATASET" \
  calibration_file=calibration/camera_top_d405_aligned.yaml \
  output_dir="runs/dp3_$RUN" \
  device=cuda:0 epochs=200 batch_size=8 val_batch_size=8
```

## Flow Matching 3D — pointcloud, nog zonder schaar-roll

Dezelfde geometrie als DP3, vier joints. Dit is rectified flow, niet de aparte
FlowPolicy-consistency-methode. Er is geen ACT-3D-trainingsscript in deze repository.

```bash
python src/greenhouse_scara_3d_flow_matching_policy/train.py \
  dataset_dir="$DATASET" \
  calibration_file=calibration/camera_top_d405_aligned.yaml \
  output_dir="runs/fm3_$RUN" \
  device=cuda:0 epochs=200 batch_size=8 val_batch_size=8
```
