#!/usr/bin/env bash
set -euo pipefail
model="${1:?Choose act3d, dp3 or fm3}"
case "$model" in
  act3d) project=greenhouse_scara_3d_act; profile=config.yaml; environment=act ;;
  dp3) project=greenhouse_scara_3d_diffusion_policy; profile=config_greenhouse.yaml; environment=diffusion_policy ;;
  fm3) project=greenhouse_scara_3d_flow_matching_policy; profile=config_greenhouse.yaml; environment=diffusion_policy ;;
  *) echo "Unknown 3D model: $model" >&2; exit 2 ;;
esac
REPO_DIR="${GREENHOUSE_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
export GREENHOUSE_DATASET_DIR="${GREENHOUSE_DATASET_DIR:-/scratch/$USER/thesis/datasets/green_house_dataset1}"
export GREENHOUSE_3D_CACHE_DIR="${GREENHOUSE_3D_CACHE_DIR:-/scratch/$USER/thesis/datasets/green_house_dataset1_pointcloud_top}"
export GREENHOUSE_MIN_EPISODE="${GREENHOUSE_MIN_EPISODE:-11}"
PIXI_PROJECT="${GREENHOUSE_PIXI_PROJECT:-/scratch/$USER/thesis/envs/$environment}"
source "$REPO_DIR/scripts/slurm_env.sh"
run_dir="$GREENHOUSE_RUN_ROOT/3d/${model}_xyzrgb_top_scissor_${SLURM_JOB_ID:-manual}"
echo "Model: $model; one camera_top; 1024 XYZRGB points; J1 J2 J3 J4 scissor_roll"
echo "Dataset: $GREENHOUSE_DATASET_DIR"
echo "Pointcloud cache: $GREENHOUSE_3D_CACHE_DIR"
echo "Run: $run_dir"
echo "Calibration: approximate intrinsics from D405 130322273198 for capture camera 130322273728"
cd "$REPO_DIR"
extra=()
if [[ "${GREENHOUSE_3D_SMOKE:-0}" == 1 ]]; then
  extra+=(epochs=1 max_train_steps=1 max_val_steps=1 batch_size=1 val_batch_size=1
          num_workers=0 use_ema=false warmup_steps=0)
fi
srun pixi run --manifest-path "$PIXI_PROJECT/pixi.toml" python -u \
  "$REPO_DIR/src/$project/train.py" --config "$REPO_DIR/src/$project/$profile" \
  "output_dir=$run_dir" "batch_size=${GREENHOUSE_3D_BATCH_SIZE:-64}" \
  "val_batch_size=${GREENHOUSE_3D_BATCH_SIZE:-64}" \
  "epochs=${GREENHOUSE_3D_EPOCHS:-100000}" \
  "num_workers=${GREENHOUSE_3D_WORKERS:-4}" device=cuda:0 "${extra[@]}"
