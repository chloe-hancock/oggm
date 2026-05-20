#!/usr/bin/env bash
#SBATCH --job-name=glacier_sa
#SBATCH --output=glacier_sa_%j.out
#SBATCH --error=glacier_sa_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node05

# DEFINE PARAMETERS 
RGI_IDS=(RGI60-06.00001 RGI60-13.00001 RGI60-14.00001)
N=100
OUTPUT_CSV_PATH='_output.csv'
PARAMS_CSV_PATH='_params.csv'

XMAX="17.0 10.0 15.0"
XMIN="1.5 0.1 -15.0"

# PATHS 
OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/oggm_20260323.sif"
RUN_SCRIPT="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/read_csvs_and_plot.py"

# Stop script on error
set -e

#######################################
# 1. Activate Conda
#######################################

echo "SLURM STARTED at $(date)"
hostname

echo "Sourcing conda..."
source /home/local/sw/miniconda/3.14/etc/profile.d/conda.sh

echo "Activating env..."
conda activate oggm_env

echo "Conda activated successfully"

# Ensure Python prints immediately
export PYTHONUNBUFFERED=1

for rid in "${RGI_IDS[@]}"; do 
    OGGM_WORKDIR="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/glacier_outs/prior/$rid" 


    # RUN THE PYTHON SCRIPT
    python "$RUN_SCRIPT" \
        --work_dir $OGGM_WORKDIR \
        --rgi_ids $rid \
        --N $N \
        --output_csv_path $OUTPUT_CSV_PATH \
        --params_csv_path $PARAMS_CSV_PATH \
        --x_max "$XMAX" \
        --x_min "$XMIN"

    echo "Finished processing $rid at $(date)"
done

echo "Job completed at $(date)" 