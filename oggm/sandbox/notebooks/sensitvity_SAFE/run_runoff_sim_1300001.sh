#!/usr/bin/env bash
#SBATCH --job-name=glacier_sa
#SBATCH --output=glacier_sa_%j.out
#SBATCH --error=glacier_sa_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node08

# === DEFINE PARAMETERS ===
MULTI_PROCESS=True   
N=10
OUTPUT_CSV_PATH='_output.csv'
PARAMS_CSV_PATH='_params.csv'
RGI_IDS='RGI60-13.00001'

XMAX="17.0 10.0 15.0"
XMIN="1.5 0.1 -15.0"

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
OGGM_WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/wd"
mkdir -p "$OGGM_WORKDIR"
export OGGM_WORKDIR
echo "Workdir for this run: $OGGM_WORKDIR"

# === PATHS ===
OGGM_IMG="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/oggm_20260323.sif"
# OGGM_WORKDIR="/home/users/chancock/glacier_outs_testing/"
RUN_SCRIPT="/home/users/chancock/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/read_csvs_and_plot.py"

# Use the local data download cache
export OGGM_DOWNLOAD_CACHE=/home/data/download
export OGGM_DOWNLOAD_CACHE_RO=1
export OGGM_EXTRACT_DIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/oggm_tmp"

# Link www chancock data here to avoid useless downloads
mkdir -p "$OGGM_WORKDIR/cache/cluster.klima.uni-bremen.de"
ln -s /home/www/chancock "$OGGM_WORKDIR/cache/cluster.klima.uni-bremen.de/~chancock"

OGGM_OUTDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/out"
export OGGM_OUTDIR
echo "Output dir for this run: $OGGM_OUTDIR"

# Stop script on error
set -e

export HOME="$OGGM_WORKDIR/fake_home"
mkdir "\$HOME"

#######################################
# 1. Activate Conda
#######################################
source /home/local/sw/miniconda/3.9/etc/profile.d/conda.sh
conda activate /home/users/chancock/.conda/envs/oggm_env

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

# Write out
echo "Copying files..."
rsync -avzh "$OGGM_OUTDIR/" output
# rsync -avz --no-perms --no-owner --no-group "$OGGM_OUTDIR/" output

echo "Job completed at $(date)"
