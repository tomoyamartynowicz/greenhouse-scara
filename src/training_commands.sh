#!/usr/bin/env bash
# Toon kopieerbare commando's; dit script start zelf geen training.
# Gebruik: bash src/training_commands.sh
cat <<'COMMANDS'
# Eenmalig uitvoeren in je terminal (pas de paden eventueel aan):
conda activate act
cd /home/tomoya/scara_ws/greenhouse-scara
DATASET=/home/tomoya/scara_ws/datasets/green_house_dataset1_from11
RUN=$(date +%Y%m%d_%H%M%S)

# DATASET bevat episodes vanaf 11; de oorspronkelijke bestanden blijven intact.
# Hieronder staat per model een zelfstandige GPU-trainingsopdracht.
# Kies het gewenste commando. Epochs en batchgrootte zijn aanpasbare voorbeelden.
# Checkpoints komen in runs/. Gebruik bij een nieuwe run een nieuwe RUN-waarde.

# 1. ACT: beide RGB-camera's (480x640), 5 joints INCLUSIEF schaar-roll.
python src/greenhouse_scara_act/imitate_episodes.py \
  --dataset_dir "$DATASET" \
  --inputs_config src/greenhouse_scara_act/inputs_rgb_dual_scissor.yaml \
  --ckpt_dir "runs/act_rgb_$RUN" \
  --device cuda --batch_size 8 --num_workers 4 \
  --num_epochs 200 --chunk_size 30 --lr 1e-5

# 2. Diffusion Policy 2D: beide RGB-camera's (480x640), 5 joints INCLUSIEF schaar-roll.
python src/greenhouse_scara_diffusion_policy/train.py \
  --config-name=train_diffusion_unet_scara_workspace \
  task=scara_rgb_dual_scissor task.dataset_path="$DATASET" \
  output_dir="$PWD/runs/dp_rgb_$RUN" run_name="dp_rgb_$RUN" \
  training.device=cuda:0 training.num_epochs=200 \
  dataloader.batch_size=8 val_dataloader.batch_size=8 logging.mode=offline

# 3. Flow Matching 2D: beide RGB-camera's (480x640), 5 dimensies INCLUSIEF schaar-roll.
python src/greenhouse_scara_flow_matching_policy/train.py \
  --config-name=train_flow_matching_unet_scara_workspace \
  task=scara_rgb_dual_scissor task.dataset_path="$DATASET" \
  output_dir="$PWD/runs/fm_rgb_$RUN" run_name="fm_rgb_$RUN" \
  training.device=cuda:0 training.num_epochs=200 \
  dataloader.batch_size=8 val_dataloader.batch_size=8 logging.mode=offline

# 3D: alleen camera_top, 1024 XYZ-punten zonder kleur; geen camerafusie.
# De calibratie hieronder hoort bij RGB-aligned depth van de bovenste D405.
# De eerste normalisatiescan en pointcloud-sampling kunnen lang duren.
# Beide 3D-modellen gebruiken nog 4 joints ZONDER schaar-roll.

# 4. Diffusion Policy 3D (DP3).
python src/greenhouse_scara_3d_diffusion_policy/train.py \
  dataset_dir="$DATASET" \
  calibration_file=calibration/camera_top_d405_aligned.yaml \
  output_dir="runs/dp3_$RUN" \
  device=cuda:0 epochs=200 batch_size=8 val_batch_size=8

# 5. Flow Matching 3D (FM3; rectified flow, niet de aparte FlowPolicy-methode).
python src/greenhouse_scara_3d_flow_matching_policy/train.py \
  dataset_dir="$DATASET" \
  calibration_file=calibration/camera_top_d405_aligned.yaml \
  output_dir="runs/fm3_$RUN" \
  device=cuda:0 epochs=200 batch_size=8 val_batch_size=8

# Extra: korte ACT-controle op CPU, beide RGB-camera's en alle 5 joints.
# Twee epochs met elk slechts 1 train- en 1 validatiebatch; geen pretraining.
OMP_NUM_THREADS=4 python src/greenhouse_scara_act/imitate_episodes.py \
  --dataset_dir "$DATASET" \
  --inputs_config src/greenhouse_scara_act/inputs_rgb_dual_scissor.yaml \
  --ckpt_dir "runs/act_rgb_smoke_$RUN" \
  --device cpu --batch_size 1 --num_workers 0 \
  --num_epochs 2 --chunk_size 30 \
  --max_train_steps 1 --max_val_steps 1 --no-pretrained_backbone

# Er is momenteel geen ACT-3D-trainingsscript in deze repository.
# Zie src/training_commands.md voor de DP-2D-test met schaar-roll en uitleg.
COMMANDS
