#!/usr/bin/env bash
# Submit from any working directory; preserve the real path through Slurm spooling.
set -euo pipefail
export GREENHOUSE_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
model="${1:?Usage: submit.sh act|act-smoke|dp|dp-smoke|fm|fm-smoke [sbatch options]}"
shift
defaults=()
case "$model" in
  act) job=src/greenhouse_scara_act/train_act.slurm ;;
  act-smoke)
    job=src/greenhouse_scara_act/train_act.slurm
    export ACT_SMOKE=1
    defaults=(--array=0 --time=00:30:00 --job-name=act_smoke)
    ;;
  dp) job=src/greenhouse_scara_diffusion_policy/train_dp.slurm ;;
  dp-smoke)
    job=src/greenhouse_scara_diffusion_policy/train_dp.slurm
    export GREENHOUSE_DP_CONFIG=train_diffusion_unet_scara_smoke
    defaults=(--time=00:30:00 --job-name=dp_smoke)
    ;;
  fm) job=src/greenhouse_scara_flow_matching_policy/train_flow_matching.slurm ;;
  fm-smoke)
    job=src/greenhouse_scara_flow_matching_policy/train_flow_matching.slurm
    export GREENHOUSE_FM_CONFIG=train_flow_matching_unet_scara_smoke
    defaults=(--time=00:30:00 --job-name=fm_smoke)
    ;;
  *) echo "Unknown model: $model (use act, act-smoke, dp, dp-smoke, fm or fm-smoke)" >&2; exit 2 ;;
esac
cd "$GREENHOUSE_REPO_DIR"
exec sbatch --export=ALL --chdir="$GREENHOUSE_REPO_DIR" "${defaults[@]}" "$@" "$job"
