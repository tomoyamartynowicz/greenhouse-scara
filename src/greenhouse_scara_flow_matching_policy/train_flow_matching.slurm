#!/bin/bash

#SBATCH --job-name=scara_fm
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=education-ME-msc-ro
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

set -euo pipefail

module purge
module load slurm

# submit.sh exports this before sbatch copies the job into its spool directory.
REPO_DIR="${GREENHOUSE_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_DIR/scripts/slurm_env.sh" ]]; then
    echo "Use scripts/submit.sh fm or submit from the repository root" >&2
    exit 1
fi
PIXI_PROJECT="${GREENHOUSE_PIXI_PROJECT:-/scratch/$USER/thesis/envs/flow_matching_policy}"
source "$REPO_DIR/scripts/slurm_env.sh"
PROJECT_DIR="$REPO_DIR/src/greenhouse_scara_flow_matching_policy"
export GREENHOUSE_FM_RUN_ROOT="${GREENHOUSE_FM_RUN_ROOT:-$GREENHOUSE_RUN_ROOT/greenhouse_scara_flow_matching_policy}"
mkdir -p "$GREENHOUSE_FM_RUN_ROOT"

cd "$PROJECT_DIR"

CONFIG_NAME="${GREENHOUSE_FM_CONFIG:-train_flow_matching_unet_scara_workspace}"
RUN_NAME="${GREENHOUSE_FM_RUN_NAME:-${CONFIG_NAME}_job${SLURM_JOB_ID}}"
echo "Dataset: $GREENHOUSE_DATASET_DIR"
echo "Config: $CONFIG_NAME"
echo "Run: $GREENHOUSE_FM_RUN_ROOT/$RUN_NAME"

srun pixi run \
    --manifest-path "$PIXI_PROJECT/pixi.toml" \
    python -u train.py \
    --config-name="$CONFIG_NAME" \
    "run_name=$RUN_NAME"
