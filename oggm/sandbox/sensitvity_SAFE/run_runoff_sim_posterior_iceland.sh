#!/usr/bin/env bash
#SBATCH --job-name=glacier_sa
#SBATCH --output=glacier_sa_%j.out
#SBATCH --error=glacier_sa_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node01

# DEFINE PARAMETERS
MULTI_PROCESS=True   
N=5000
OUTPUT_CSV_PATH='_output.csv'
PARAMS_CSV_PATH='_params.csv'

# Iceland icecap RGI IDs
RGI_IDS=(RGI60-06.00234 
RGI60-06.00237 
RGI60-06.00232 
RGI60-06.00236 
RGI60-06.00235
RGI60-06.00238
RGI60-06.00228
RGI60-06.00233
RGI60-06.00229
RGI60-06.00231)

# PATHS
# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
OGGM_WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/wd"
mkdir -p "$OGGM_WORKDIR"
export OGGM_WORKDIR
echo "Workdir for this run: $OGGM_WORKDIR"

OGGM_OUTDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/out"
mkdir -p "$OGGM_OUTDIR"
export OGGM_OUTDIR
echo "Output dir for this run: $OGGM_OUTDIR"

OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/oggm_20260323.sif"
RUN_SCRIPT="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/runoff_sim.py"

# Stop script on error
set -e

#######################################
# 1. Activate Conda
#######################################
source /home/local/sw/miniconda/3.14/etc/profile.d/conda.sh
conda activate oggm_env

# Ensure Python prints immediately
export PYTHONUNBUFFERED=1

mkdir -p glacier_outs/posterior

for rid in "${RGI_IDS[@]}"; do 
    CSV="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/glacier_outs/prior/$rid/reduced_bounds.csv"
        
    line=$(sed -n '2p' "$CSV")
    IFS=',' read -r xmin1 xmin2 xmin3 xmax1 xmax2 xmax3 <<< "$line"
    
    XMAX="$xmax1 $xmax2 $xmax3"
    XMIN="$xmin1 $xmin2 $xmin3"

    OGGM_WORKDIR="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/glacier_outs/posterior/$rid" 

    # RUN THE PYTHON SCRIPT
    python "$RUN_SCRIPT" \
        --work_dir $OGGM_WORKDIR \
        --out_dir $OGGM_OUTDIR \
        --rgi_ids $rid \
        --N $N \
        --output_csv_path $OUTPUT_CSV_PATH \
        --params_csv_path $PARAMS_CSV_PATH \
        --x_max $xmax1 $xmax2 $xmax3 \
        --x_min $xmin1 $xmin2 $xmin3

    # Write out
    echo "Copying files..."

    rsync -avzh "$OGGM_OUTDIR/" glacier_outs/posterior/$rid/
done

# rsync -avz --no-perms --no-owner --no-group "$OGGM_OUTDIR/" output

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"
