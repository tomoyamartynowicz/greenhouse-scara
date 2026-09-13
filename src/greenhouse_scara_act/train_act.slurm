#!/bin/bash

#SBATCH --job-name=act_chunk_sweep
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=education-ME-msc-ro
#SBATCH --array=0-5
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

module purge
module load slurm

# submit.sh exports this before sbatch copies the job into its spool directory.
REPO_DIR="${GREENHOUSE_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_DIR/scripts/slurm_env.sh" ]]; then
    echo "Use scripts/submit.sh act or submit from the repository root" >&2
    exit 1
fi
PIXI_PROJECT="${GREENHOUSE_PIXI_PROJECT:-/scratch/$USER/thesis/envs/act}"
source "$REPO_DIR/scripts/slurm_env.sh"
PROJECT_DIR="$REPO_DIR/src/greenhouse_scara_act"
DATASET_DIR="$GREENHOUSE_DATASET_DIR"

# ------------------------------------------------------------
# Experiment matrix
#
# 0 -> batch 64,  chunk 15
# 1 -> batch 64,  chunk 30
# 2 -> batch 64,  chunk 45
# 3 -> batch 128, chunk 15
# 4 -> batch 128, chunk 30
# 5 -> batch 128, chunk 45
# ------------------------------------------------------------

BATCH_SIZES=(64 64 64 128 128 128)
CHUNK_SIZES=(15 30 45 15 30 45)

IDX="$SLURM_ARRAY_TASK_ID"

BATCH_SIZE="${BATCH_SIZES[$IDX]}"
CHUNK_SIZE="${CHUNK_SIZES[$IDX]}"

LR="5e-5"
KL_WEIGHT=10
NUM_EPOCHS=5000

RUN_NAME="bs${BATCH_SIZE}_chunk${CHUNK_SIZE}_lr${LR}_kl${KL_WEIGHT}"

CKPT_DIR="${GREENHOUSE_ACT_RUN_ROOT:-$GREENHOUSE_RUN_ROOT/greenhouse_scara_act}/chunk_sweep/$RUN_NAME"

mkdir -p "$CKPT_DIR"
mkdir -p "$TORCH_HOME"

# Store all stdout/stderr with this particular experiment.
LOG_FILE="$CKPT_DIR/train_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG_FILE" 2>&1

cd "$PROJECT_DIR"

echo "=================================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Array job:      $SLURM_ARRAY_JOB_ID"
echo "Array task:     $SLURM_ARRAY_TASK_ID"
echo "Node:           $(hostname)"
echo
echo "Batch size:     $BATCH_SIZE"
echo "Chunk size:     $CHUNK_SIZE"
echo "Chunk duration: $(awk "BEGIN { print $CHUNK_SIZE / 30 }") s @ 30 Hz"
echo "Learning rate:  $LR"
echo "KL weight:      $KL_WEIGHT"
echo "Epochs:         $NUM_EPOCHS"
echo
echo "Dataset:        $DATASET_DIR"
echo "Checkpoint dir: $CKPT_DIR"
echo "Started:        $(date)"
echo "=================================================="

srun pixi run \
  --manifest-path "$PIXI_PROJECT/pixi.toml" \
  python -u imitate_episodes.py \
  --dataset_dir "$DATASET_DIR" \
  --ckpt_dir "$CKPT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --chunk_size "$CHUNK_SIZE" \
  --lr "$LR" \
  --kl_weight "$KL_WEIGHT" \
  --num_epochs "$NUM_EPOCHS"

echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
