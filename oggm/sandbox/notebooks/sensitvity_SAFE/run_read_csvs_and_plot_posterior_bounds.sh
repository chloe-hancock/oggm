#!/usr/bin/env bash
#SBATCH --job-name=glacier_sa
#SBATCH --output=glacier_sa_%j.out
#SBATCH --error=glacier_sa_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node08

# DEFINE PARAMETERS 
RGI_IDS=(RGI60-06.00001 RGI60-13.00001 RGI60-14.00001)
N=100
OUTPUT_CSV_PATH='_output.csv'
PARAMS_CSV_PATH='_params.csv'

# PATHS 
OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/oggm_20260323.sif"
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

for rid in "${RGI_IDS[@]}"; do 
    CSV="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/glacier_outs/prior/$rid/reduced_bounds.csv"
    
    line=$(sed -n '2p' "$CSV")
    IFS=',' read -r glacier xmin1 xmin2 xmin3 xmax1 xmax2 xmax3 <<< "$line"
    
    XMAX="$xmax1 $xmax2 $xmax3"
    XMIN="$xmin1 $xmin2 $xmin3"

    OGGM_WORKDIR="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/glacier_outs/posterior/$rid" 

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