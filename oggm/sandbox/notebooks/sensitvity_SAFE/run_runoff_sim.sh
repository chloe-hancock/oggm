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
RUN_SCRIPT="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/runoff_sim.py"

# === DEFINE PARAMETERS ===
BORDER=10
STORE_MODEL_GEOM=True
MIN_ICE_THICK_FOR_LEN=1
RGI_IDS='RGI60-14.00063'
MULTI_PROCESS=True
BASE_URL='https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup'    
N=100
YR_START=1901
YR_END=2020
SPINUP_PERIOD=95
OUTPUT_CSV_PATH='_pakistan_runoff_output.csv'
PARAMS_CSV_PATH='_pakistan_params.csv'

# Stop script on error
set -e

#######################################
# 1. Activate Conda
#######################################
source /home/local/sw/miniconda/3.9/etc/profile.d/conda.sh
conda activate oggm_docs

# Ensure Python prints immediately
export PYTHONUNBUFFERED=1

# === RUN THE PYTHON SCRIPT ===
python "$RUN_SCRIPT" \
    --work_dir $OGGM_WORKDIR \
    --border $BORDER \
    --store_model_geom $STORE_MODEL_GEOM \
    --min_ice_thick $MIN_ICE_THICK_FOR_LEN \
    --rgi_ids $RGI_IDS \
    --multi_process $MULTI_PROCESS \
    --base_url $BASE_URL \
    --N $N \
    --year_start $YR_START \
    --year_end $YR_END \
    --spinup_period $SPINUP_PERIOD \
    --output_csv_path $OUTPUT_CSV_PATH \
    --params_csv_path $PARAMS_CSV_PATH

echo "Job completed at $(date)"
