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
RGI_IDS=RGI60-06.00001

XMIN="1.5 0.1 -15.0" # Minimum values for each parameter
XMAX="17.0 10.0 15.0" # Maximum values for each parameter

# XMAX="2.55278266  0.81903717 -0.78842941"
# XMIN="1.64705155  0.58564001 -2.29988439"

# === PATHS ===
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

OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/oggm_20260323.sif"
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
    --out_dir $OGGM_OUTDIR \
    --rgi_ids $RGI_IDS \
    --N $N \
    --output_csv_path $OUTPUT_CSV_PATH \
    --params_csv_path $PARAMS_CSV_PATH \
    --x_max "$XMAX" \
    --x_min "$XMIN"

# Write out

echo "Copying files..."
mkdir -p glacier_outs
rsync -avzh "$OGGM_OUTDIR/" glacier_outs/$RGI_IDS"_boundaries_test"
# rsync -avz --no-perms --no-owner --no-group "$OGGM_OUTDIR/" output

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"
