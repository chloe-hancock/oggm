#!/usr/bin/env bash
#SBATCH --job-name=glacier_sa
#SBATCH --output=glacier_sa_%j.out
#SBATCH --error=glacier_sa_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node01

# === DEFINE PARAMETERS ===
MULTI_PROCESS=True   
N=5000
OUTPUT_CSV_PATH='_output.csv'
PARAMS_CSV_PATH='_params.csv'
RGI_IDS='RGI60-06.00001'

XMAX="17.0 10.0 15.0"
XMIN="1.5 0.1 -15.0"

# === PATHS ===
OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/oggm_20260323.sif"
OGGM_WORKDIR="/home/users/chancock/glacier_outs/"$RGI_IDS
RUN_SCRIPT="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/runoff_sim.py"

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
python "$RUN_SCRIPT" \
    --work_dir $OGGM_WORKDIR \
    --rgi_ids $RGI_IDS \
    --N $N \
    --output_csv_path $OUTPUT_CSV_PATH \
    --params_csv_path $PARAMS_CSV_PATH \
    --x_max "$XMAX" \
    --x_min "$XMIN"

echo "Job completed at $(date)"
