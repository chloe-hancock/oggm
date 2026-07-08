#!/usr/bin/env bash
#SBATCH --job-name=glacier_sa
#SBATCH --output=glacier_sa_%j.out
#SBATCH --error=glacier_sa_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node24

# DEFINE PARAMETERS 
# RGI_IDS=(RGI60-06.00001 RGI60-13.00001 RGI60-14.00001)
# RGI_IDS=(RGI60-06.00377 RGI60-11.00897 RGI60-13.53223)
RGI_IDS=(RGI60-06.00377 RGI60-13.53223)
N=5000
OUTPUT_CSV_PATH='_output.csv'
PARAMS_CSV_PATH='_params.csv'

# PATHS 
OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/oggm_20260323.sif"
RUN_SCRIPT="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/read_csvs_and_plot.py"

area_uncertainty_south_asia_west=7.7

area_uncertainty_iceland=2.6
area_uncertainty_central_europe=10.4
area_uncertainty_central_asia=8.4

# AREA_UNCERTAINTY_LIST=(
#     "$area_uncertainty_iceland"
#     "$area_uncertainty_central_europe"
#     "$area_uncertainty_central_asia"
# )

AREA_UNCERTAINTY_LIST=(
    "$area_uncertainty_iceland"
    "$area_uncertainty_central_asia"
)


# Stop script on error
set -e

#######################################
# 1. Activate Conda
#######################################
source /home/local/sw/miniconda/3.14/etc/profile.d/conda.sh
conda activate oggm_env

# Ensure Python prints immediately
export PYTHONUNBUFFERED=1

for i in "${!RGI_IDS[@]}"; do 

    rid="${RGI_IDS[$i]}"
    area_unc="${AREA_UNCERTAINTY_LIST[$i]}"

    echo "Processing $rid with area uncertainty $area_unc"

    CSV="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/glacier_outs/prior/$rid/reduced_bounds.csv"
    
    line=$(sed -n '2p' "$CSV")
    IFS=',' read -r xmin1 xmin2 xmin3 xmin4 xmin5 xmin6 xmax1 xmax2 xmax3 xmax4 xmax5 xmax6 <<< "$line"

    OGGM_WORKDIR="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/glacier_outs/posterior/$rid" 

    # RUN THE PYTHON SCRIPT
    python "$RUN_SCRIPT" \
        --work_dir $OGGM_WORKDIR \
        --rgi_ids $rid \
        --N $N \
        --output_csv_path $OUTPUT_CSV_PATH \
        --params_csv_path $PARAMS_CSV_PATH \
        --x_max $xmax1 $xmax2 $xmax3 $xmax4 $xmax5 $xmax6 \
        --x_min $xmin1 $xmin2 $xmin3 $xmin4 $xmin5 $xmin6 \
        --area_uncertainty $area_unc

    echo "Finished processing $rid at $(date)"
done

echo "Job completed at $(date)" 
