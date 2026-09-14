#!/bin/bash
#SBATCH --job-name=scara_fm3
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=education-ME-msc-ro
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
set -euo pipefail
module purge
module load slurm
REPO_DIR="${GREENHOUSE_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PIXI_PROJECT="${GREENHOUSE_PIXI_PROJECT:-/scratch/$USER/thesis/envs/diffusion_policy}"
# Smoke uses explicit synthetic two-camera geometry; normal training uses real calibration.
PROFILE="${GREENHOUSE_FM3_CONFIG:-config.yaml}"
DEFAULT_DATASET=test_rgbd_single_joints
DEFAULT_CALIBRATION=camera_top_d405_aligned.yaml
if [[ "${GREENHOUSE_FM3_SMOKE:-0}" == 1 && -z "${GREENHOUSE_FM3_CONFIG:-}" ]]; then
    PROFILE=config_dummy.yaml
fi
if [[ "$PROFILE" == config_dummy.yaml ]]; then
    DEFAULT_DATASET=greenhouse_dummy_dataset
    DEFAULT_CALIBRATION=dummy_two_cameras.yaml
fi
export GREENHOUSE_DATASET_DIR="${GREENHOUSE_FM3_DATASET_DIR:-/scratch/$USER/thesis/datasets/$DEFAULT_DATASET}"
source "$REPO_DIR/scripts/slurm_env.sh"
CALIBRATION="${GREENHOUSE_FM3_CALIBRATION:-$REPO_DIR/calibration/$DEFAULT_CALIBRATION}"
RUN_ROOT="${GREENHOUSE_FM3_RUN_ROOT:-$GREENHOUSE_RUN_ROOT/greenhouse_scara_3d_flow_matching_policy}"
ARGS=()
if [[ "${GREENHOUSE_FM3_SMOKE:-0}" == 1 ]]; then
    ARGS+=(allow_single_episode=true epochs=1 max_train_steps=1 max_val_steps=1
           batch_size=1 val_batch_size=1 use_ema=false warmup_steps=0)
fi
echo "Dataset: $GREENHOUSE_DATASET_DIR"
echo "Calibration: $CALIBRATION"
echo "Output: $RUN_ROOT/job${SLURM_JOB_ID}"
cd "$REPO_DIR"
srun pixi run --manifest-path "$PIXI_PROJECT/pixi.toml" \
    python -u src/greenhouse_scara_3d_flow_matching_policy/train.py \
    --config "$REPO_DIR/src/greenhouse_scara_3d_flow_matching_policy/$PROFILE" \
    "dataset_dir=$GREENHOUSE_DATASET_DIR" "calibration_file=$CALIBRATION" \
    "output_dir=$RUN_ROOT/job${SLURM_JOB_ID}" device=cuda:0 "${ARGS[@]}"
