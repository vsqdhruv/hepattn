#!/bin/bash

#########################
# SLURM JOB CONFIGURATION
#########################

# allocate CPU partition
#SBATCH --partition=COMPUTE

# job name
#SBATCH --job-name=pytest_nodata

# request 1 node
#SBATCH --nodes=1

# request 1 cpu core
#SBATCH --ntasks=1

# request memory for job
#SBATCH --mem=2G

# maximum runtime for job
#SBATCH --time=00:20:00

# email notis
#SBATCH --mail-user=zcapnai@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

# output files go to slurm_logs folder
#SBATCH --output=/home/xzcapnai/hepattn/hepattn/slurm_logs/pytest_nodata_%j.out
#SBATCH --error=/home/xzcapnai/hepattn/hepattn/slurm_logs/pytest_nodata_%j.err

###############
# JOB EXECUTION
###############

# create log folder if it doesnt alr exist
mkdir -p /home/xzcapnai/hepattn/hepattn/slurm_logs

# move to project directory
cd /home/xzcapnai/hepattn/hepattn

# echo debug info
echo "=== Starting pytest (not gpu and not requiresdata) ==="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Current directory: $(pwd)"

# run pytest inside Apptainer container w/ pixi environment
srun apptainer exec --bind /home/xzcapnai/hepattn/hepattn \
	pixi.sif pixi run pytest -m "not gpu and not requiresdata"

# Finished
echo "=== Pytest job completed at $(date) ==="
