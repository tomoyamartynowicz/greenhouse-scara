# Sourced by the model jobs, after resolving GREENHOUSE_REPO_DIR.
export PATH="$HOME/.pixi/bin:$PATH"
export GREENHOUSE_DATASET_DIR="${GREENHOUSE_DATASET_DIR:-/scratch/$USER/thesis/datasets/greenhouse_dummy_dataset}"
export GREENHOUSE_RUN_ROOT="${GREENHOUSE_RUN_ROOT:-/scratch/$USER/thesis/runs}"
export TORCH_HOME="${TORCH_HOME:-/scratch/$USER/thesis/cache/torch}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ ! -d "$GREENHOUSE_DATASET_DIR" ]]; then
    echo "Dataset not found: $GREENHOUSE_DATASET_DIR" >&2
    exit 1
fi
if [[ ! -f "$PIXI_PROJECT/pixi.toml" ]]; then
    echo "Pixi manifest not found: $PIXI_PROJECT/pixi.toml (set GREENHOUSE_PIXI_PROJECT)" >&2
    exit 1
fi
mkdir -p "$TORCH_HOME" "$GREENHOUSE_RUN_ROOT"
