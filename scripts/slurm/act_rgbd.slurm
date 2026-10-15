#!/usr/bin/env bash
#SBATCH --job-name=act_rgbd_scissor
#SBATCH --partition=gpu
#SBATCH --account=education-ME-msc-ro
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
set -euo pipefail
module purge
module load slurm
REPO_DIR="${GREENHOUSE_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$REPO_DIR/scripts/train_2d_job.sh" act rgbd
