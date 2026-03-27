#!/usr/bin/env bash
#SBATCH --job-name=glacier_sa
#SBATCH --output=glacier_sa_%j.out
#SBATCH --error=glacier_sa_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# === PATHS ===
OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/oggm_20260323.sif"
OGGM_WORKDIR="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/glacier_outs"
RUN_SCRIPT="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/read_csvs_and_plot.py"

# Stop script on error
set -e

#######################################
# 1. Activate Conda
#######################################
source /home/local/sw/miniconda/3.9/etc/profile.d/conda.sh
conda activate oggm_env

# Ensure Python prints immediately
export PYTHONUNBUFFERED=1

# === RUN THE PYTHON SCRIPT ===
python "$RUN_SCRIPT"

echo "Job completed at $(date)"
