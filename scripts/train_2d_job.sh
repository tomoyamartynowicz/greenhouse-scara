#!/usr/bin/env bash
# Called by one of the six scripts/slurm/*_*.slurm files.
set -euo pipefail
model="${1:?model required}"
modality="${2:?modality required}"
case "$model:$modality" in
  act:rgb|act:rgbd|dp:rgb|dp:rgbd|fm:rgb|fm:rgbd) ;;
  *) echo "Unknown 2D experiment: $model:$modality" >&2; exit 2 ;;
esac
REPO_DIR="${GREENHOUSE_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
export GREENHOUSE_DATASET_DIR="${GREENHOUSE_DATASET_DIR:-/scratch/$USER/thesis/datasets/green_house_dataset1}"
export GREENHOUSE_MIN_EPISODE="${GREENHOUSE_MIN_EPISODE:-11}"
if [[ ! "$GREENHOUSE_MIN_EPISODE" =~ ^[0-9]+$ ]]; then
  echo "GREENHOUSE_MIN_EPISODE must be a nonnegative integer" >&2
  exit 2
fi
if [[ "$model" == act ]]; then
  PIXI_PROJECT="${GREENHOUSE_PIXI_PROJECT:-/scratch/$USER/thesis/envs/act}"
else
  PIXI_PROJECT="${GREENHOUSE_PIXI_PROJECT:-/scratch/$USER/thesis/envs/diffusion_policy}"
fi
source "$REPO_DIR/scripts/slurm_env.sh"
shopt -s nullglob
episodes=()
for episode in "$GREENHOUSE_DATASET_DIR"/episode_*.hdf5; do
  episode_name="${episode##*/}"
  if [[ "$episode_name" =~ ^episode_([0-9]+)\.hdf5$ ]] && \
     (( 10#${BASH_REMATCH[1]} >= GREENHOUSE_MIN_EPISODE )); then
    episodes+=("$episode")
  fi
done
if (( ${#episodes[@]} < 2 )); then
  echo "Expected at least two episode_*.hdf5 files in $GREENHOUSE_DATASET_DIR" >&2
  exit 1
fi
for episode in "${episodes[@]}"; do
  if [[ ! -f "$episode" ]]; then
    echo "Broken dataset symlink or unreadable file: $episode" >&2
    exit 1
  fi
done
run_name="${model}_${modality}_scissor_${SLURM_JOB_ID:-manual}"
run_dir="$GREENHOUSE_RUN_ROOT/2d/$run_name"
mkdir -p "$run_dir"
echo "Dataset: $GREENHOUSE_DATASET_DIR (${#episodes[@]} episodes from $GREENHOUSE_MIN_EPISODE)"
echo "Run: $run_dir"
echo "Pixi: $PIXI_PROJECT"
echo "Model: $model; input: $modality; actions: J1 J2 J3 J4 scissor_roll"
cd "$REPO_DIR"

if [[ "$model" == act ]]; then
  inputs="$REPO_DIR/src/greenhouse_scara_act/inputs_${modality}_dual_scissor.yaml"
  batch_default=2
  [[ "$modality" == rgbd ]] && batch_default=1
  extra=()
  if [[ "${ACT_PRETRAINED_BACKBONE:-1}" == 0 ]]; then extra+=(--no-pretrained_backbone); fi
  srun pixi run --manifest-path "$PIXI_PROJECT/pixi.toml" python -u \
    "$REPO_DIR/src/greenhouse_scara_act/imitate_episodes.py" \
    --dataset_dir "$GREENHOUSE_DATASET_DIR" --inputs_config "$inputs" \
    --ckpt_dir "$run_dir" --device cuda \
    --batch_size "${ACT_BATCH_SIZE:-$batch_default}" --num_workers "${ACT_NUM_WORKERS:-4}" \
    --num_epochs "${ACT_NUM_EPOCHS:-200}" --chunk_size "${ACT_CHUNK_SIZE:-30}" \
    --lr "${ACT_LR:-1e-5}" "${extra[@]}"
else
  if [[ "$model" == dp ]]; then
    project="$REPO_DIR/src/greenhouse_scara_diffusion_policy"
    config=train_diffusion_unet_scara_workspace
    export GREENHOUSE_DP_RUN_ROOT="$GREENHOUSE_RUN_ROOT/2d"
  else
    project="$REPO_DIR/src/greenhouse_scara_flow_matching_policy"
    config=train_flow_matching_unet_scara_workspace
    export GREENHOUSE_FM_RUN_ROOT="$GREENHOUSE_RUN_ROOT/2d"
  fi
  batch_default=4
  [[ "$modality" == rgbd ]] && batch_default=2
  cd "$project"
  srun pixi run --manifest-path "$PIXI_PROJECT/pixi.toml" python -u train.py \
    --config-name="$config" task="scara_${modality}_dual_scissor" \
    "run_name=$run_name" "output_dir=$run_dir" \
    "training.num_epochs=${GREENHOUSE_2D_EPOCHS:-200}" \
    "dataloader.batch_size=${GREENHOUSE_2D_BATCH_SIZE:-$batch_default}" \
    "val_dataloader.batch_size=${GREENHOUSE_2D_BATCH_SIZE:-$batch_default}" \
    logging.mode=offline
fi
