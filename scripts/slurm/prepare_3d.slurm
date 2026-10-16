#!/usr/bin/env bash
#SBATCH --job-name=prepare_xyzrgb_top
#SBATCH --partition=compute
#SBATCH --account=education-ME-msc-ro
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
set -euo pipefail
module purge
module load slurm
REPO_DIR="${GREENHOUSE_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export GREENHOUSE_DATASET_DIR="${GREENHOUSE_DATASET_DIR:-/scratch/$USER/thesis/datasets/green_house_dataset1}"
export GREENHOUSE_3D_CACHE_DIR="${GREENHOUSE_3D_CACHE_DIR:-/scratch/$USER/thesis/datasets/green_house_dataset1_pointcloud_top}"
export GREENHOUSE_MIN_EPISODE="${GREENHOUSE_MIN_EPISODE:-11}"
PIXI_PROJECT="${GREENHOUSE_PIXI_PROJECT:-/scratch/$USER/thesis/envs/diffusion_policy}"
source "$REPO_DIR/scripts/slurm_env.sh"
cd "$REPO_DIR"
srun pixi run --manifest-path "$PIXI_PROJECT/pixi.toml" python -u \
  scripts/prepare_3d.py --workers "${SLURM_CPUS_PER_TASK:-4}"
