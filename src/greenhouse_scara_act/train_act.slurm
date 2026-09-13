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
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.out

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
INPUTS_CONFIG="${GREENHOUSE_ACT_INPUTS_CONFIG:-$PROJECT_DIR/inputs_rgb_dual.yaml}"
if [[ ! -f "$INPUTS_CONFIG" ]]; then
    echo "Input config not found: $INPUTS_CONFIG" >&2
    exit 1
fi

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
LRS=(5e-5 5e-5 5e-5 1e-4 1e-4 1e-4)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "$IDX" =~ ^[0-5]$ ]]; then
    echo "Array index must be 0..5" >&2
    exit 1
fi

BATCH_SIZE="${ACT_BATCH_SIZE:-${BATCH_SIZES[$IDX]}}"
CHUNK_SIZE="${ACT_CHUNK_SIZE:-${CHUNK_SIZES[$IDX]}}"

LR="${ACT_LR:-${LRS[$IDX]}}"
KL_WEIGHT="${ACT_KL_WEIGHT:-10}"
NUM_EPOCHS="${ACT_NUM_EPOCHS:-5000}"
EXTRA_ARGS=()
RUN_GROUP=chunk_sweep
if [[ "${ACT_SMOKE:-0}" == 1 ]]; then
    BATCH_SIZE=2
    NUM_EPOCHS=1
    RUN_GROUP=smoke
    EXTRA_ARGS+=(--max_train_steps 1 --max_val_steps 1 --num_workers 0)
fi
if [[ "${ACT_PRETRAINED_BACKBONE:-1}" == 0 ]]; then
    EXTRA_ARGS+=(--no-pretrained_backbone)
fi

RUN_NAME="bs${BATCH_SIZE}_chunk${CHUNK_SIZE}_lr${LR}_kl${KL_WEIGHT}_job${SLURM_JOB_ID}_${IDX}"

CKPT_DIR="${GREENHOUSE_ACT_RUN_ROOT:-$GREENHOUSE_RUN_ROOT/greenhouse_scara_act}/$RUN_GROUP/$RUN_NAME"

mkdir -p "$CKPT_DIR"
mkdir -p "$TORCH_HOME"

# Store all stdout/stderr with this particular experiment.
LOG_FILE="$CKPT_DIR/train_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${IDX}.log"
exec > "$LOG_FILE" 2>&1

cd "$PROJECT_DIR"

echo "=================================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Array job:      ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
echo "Array task:     $IDX"
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
echo "Input config:   $INPUTS_CONFIG"
echo "Test mode:      ${ACT_SMOKE:-0}"
echo "Checkpoint dir: $CKPT_DIR"
echo "Started:        $(date)"
echo "=================================================="

srun pixi run \
  --manifest-path "$PIXI_PROJECT/pixi.toml" \
  python -u imitate_episodes.py \
  --dataset_dir "$DATASET_DIR" \
  --inputs_config "$INPUTS_CONFIG" \
  --device cuda \
  --ckpt_dir "$CKPT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --chunk_size "$CHUNK_SIZE" \
  --lr "$LR" \
  --kl_weight "$KL_WEIGHT" \
  --num_epochs "$NUM_EPOCHS" \
  "${EXTRA_ARGS[@]}"

echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
