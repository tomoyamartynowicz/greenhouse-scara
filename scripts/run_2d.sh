#!/usr/bin/env bash
# Offline inference on recorded episodes; no robot motion.
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
  echo "Usage: bash scripts/run_2d.sh act|dp|fm rgb|rgbd CHECKPOINT [DATASET_DIR]" >&2
  exit 2
fi
model="$1"
modality="$2"
checkpoint="$3"
dataset="${4:-${GREENHOUSE_DATASET_DIR:-}}"
export GREENHOUSE_MIN_EPISODE="${GREENHOUSE_MIN_EPISODE:-11}"
case "$model:$modality" in
  act:rgb|act:rgbd|dp:rgb|dp:rgbd|fm:rgb|fm:rgbd) ;;
  *) echo "Unknown experiment: $model:$modality" >&2; exit 2 ;;
esac
if [[ -z "$dataset" || ! -d "$dataset" ]]; then
  echo "Set DATASET_DIR to the copied green_house_dataset1_from11 directory" >&2
  exit 1
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 1
fi
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
checkpoint="$(realpath "$checkpoint")"
dataset="$(realpath "$dataset")"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$model" == act ]]; then
  export PYTHONPATH="$repo/src/greenhouse_scara_act:$PYTHONPATH"
  exec python "$repo/src/greenhouse_scara_act/eval_offline.py" \
    --ckpt-dir "$(dirname "$checkpoint")" --checkpoint "$checkpoint" \
    --dataset-dir "$dataset" --device "${GREENHOUSE_INFERENCE_DEVICE:-cpu}" \
    --samples "${GREENHOUSE_EVAL_SAMPLES:-16}" --expected-modality "$modality"
fi
if [[ "$model" == dp ]]; then
  project=greenhouse_scara_diffusion_policy
else
  project=greenhouse_scara_flow_matching_policy
fi
export PYTHONPATH="$repo/src/$project:$PYTHONPATH"
output="${GREENHOUSE_EVAL_DIR:-$repo/runs/offline_eval/${model}_${modality}_$(date +%Y%m%d_%H%M%S)_$$}"
extra=(--samples "${GREENHOUSE_EVAL_SAMPLES:-16}" --expected-modality "$modality")
if [[ -n "${GREENHOUSE_INFERENCE_STEPS:-}" ]]; then
  extra+=(--num-inference-steps "$GREENHOUSE_INFERENCE_STEPS")
fi
exec python "$repo/src/$project/eval.py" \
  --checkpoint "$checkpoint" --output_dir "$output" \
  --dataset-dir "$dataset" --device "${GREENHOUSE_INFERENCE_DEVICE:-cpu}" \
  "${extra[@]}"
