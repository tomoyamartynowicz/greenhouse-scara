#!/bin/bash
#SBATCH --job-name=scara_dp3
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=education-ME-msc-ro
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
set -euo pipefail
module purge
module load slurm
REPO_DIR="${GREENHOUSE_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PIXI_PROJECT="${GREENHOUSE_PIXI_PROJECT:-/scratch/$USER/thesis/envs/diffusion_policy}"
# This real one-camera demo needs its actual calibration, not the 2D dummy data.
export GREENHOUSE_DATASET_DIR="${GREENHOUSE_DP3_DATASET_DIR:-/scratch/$USER/thesis/datasets/test_rgbd_single_joints}"
source "$REPO_DIR/scripts/slurm_env.sh"
CALIBRATION="${GREENHOUSE_DP3_CALIBRATION:-$REPO_DIR/calibration/camera_top_d405_aligned.yaml}"
RUN_ROOT="${GREENHOUSE_DP3_RUN_ROOT:-$GREENHOUSE_RUN_ROOT/greenhouse_scara_3d_diffusion_policy}"
ARGS=()
if [[ "${GREENHOUSE_DP3_SMOKE:-0}" == 1 ]]; then
    ARGS+=(allow_single_episode=true epochs=1 max_train_steps=1 max_val_steps=1
           batch_size=1 val_batch_size=1 use_ema=false warmup_steps=0)
fi
echo "Dataset: $GREENHOUSE_DATASET_DIR"
echo "Calibration: $CALIBRATION"
echo "Output: $RUN_ROOT/job${SLURM_JOB_ID}"
cd "$REPO_DIR"
srun pixi run --manifest-path "$PIXI_PROJECT/pixi.toml" \
    python -u src/greenhouse_scara_3d_diffusion_policy/train.py \
    "dataset_dir=$GREENHOUSE_DATASET_DIR" "calibration_file=$CALIBRATION" \
    "output_dir=$RUN_ROOT/job${SLURM_JOB_ID}" device=cuda:0 "${ARGS[@]}"
