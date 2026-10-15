#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: bash scripts/submit_2d.sh act|dp|fm rgb|rgbd [sbatch options]" >&2
  exit 2
fi
model="$1"
modality="$2"
shift 2
case "$model:$modality" in
  act:rgb|act:rgbd|dp:rgb|dp:rgbd|fm:rgb|fm:rgbd) ;;
  *) echo "Unknown experiment: $model:$modality" >&2; exit 2 ;;
esac
export GREENHOUSE_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$GREENHOUSE_REPO_DIR"
exec sbatch --export=ALL --chdir="$GREENHOUSE_REPO_DIR" "$@" \
  "scripts/slurm/${model}_${modality}.slurm"
