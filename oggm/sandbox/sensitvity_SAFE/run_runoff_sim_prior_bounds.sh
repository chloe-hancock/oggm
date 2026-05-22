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
# RGI_IDS=(RGI60-06.00001 RGI60-13.00001 RGI60-14.00001)
# RGI_IDS=(RGI60-06.00001)
RGI_IDS=(RGI60-13.00001 RGI60-14.00001)

# Minimum values for each parameter
xmin1=1.5
xmin2=0.1
xmin3=-15.0

# Maximum values for each parameter
xmax1=17.0
xmax2=10.0
xmax3=15.0

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

for rid in "${RGI_IDS[@]}"; do 
    # OGGM_WORKDIR="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/sensitvity_SAFE/glacier_outs/prior/$rid" 

    # PATHS
    # On every node, when slurm starts a job, it will make sure the directory
    # /work/username exists and is writable by the jobs user.
    # We create a sub-directory there for this job to store its runtime data at.
    OGGM_WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/wd/$rid"
    mkdir -p "$OGGM_WORKDIR"
    export OGGM_WORKDIR
    echo "Workdir for this run: $OGGM_WORKDIR"  

    OGGM_OUTDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/out/$rid"
    mkdir -p "$OGGM_OUTDIR"
    export OGGM_OUTDIR
    echo "Output dir for this run: $OGGM_OUTDIR"

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
    mkdir -p glacier_outs/prior

    rsync -avzh "$OGGM_OUTDIR/" glacier_outs/prior/$rid/
    done

# rsync -avz --no-perms --no-owner --no-group "$OGGM_OUTDIR/" output

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"
