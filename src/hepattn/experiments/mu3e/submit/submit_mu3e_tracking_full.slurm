#!/bin/bash

#SBATCH --job-name=mu3e-train-tracking
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=/home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/mu3e/slurm_logs/slurm-%j.%x.out
#SBATCH --time=24:00:00

# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Move to workdir
cd /home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/mu3e/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

echo "nvidia-smi:"
nvidia-smi

# Run the training
echo "Running training script..."

# Python command that will be run
PYTORCH_CMD="python run_tracking.py fit --config configs/mu3e_tracking.yaml"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="srun apptainer run --nv --bind /home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/mu3e \
						 /home/xzcapnai/hepattn/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"