#!/usr/bin/env bash
#SBATCH --job-name=glacier_sa
#SBATCH --output=glacier_sa_%j.out
#SBATCH --error=glacier_sa_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node09

# === DEFINE PARAMETERS ===
RGI_IDS=RGI60-06.00001
N=5000
OUTPUT_CSV_PATH='_output.csv'
PARAMS_CSV_PATH='_params.csv'

# XMAX="17.0 10.0 15.0"
# XMIN="1.5 0.1 -15.0"

# After
# XMAX="3.27491219  0.86718002 -2.88411379"
# XMIN="2.27491219 -0.13281998 -3.88411379"

# Before
XMAX="2.55278266  0.81903717 -0.78842941"
XMIN="1.64705155  0.58564001 -2.29988439"

# === PATHS ===
OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/oggm_20260323.sif"
OGGM_WORKDIR="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/glacier_outs/"$RGI_IDS
RUN_SCRIPT="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/read_csvs_and_plot.py"

# Stop script on error
set -e

#######################################
# 1. Activate Conda
#######################################
source /home/local/sw/miniconda/3.14/etc/profile.d/conda.sh

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
