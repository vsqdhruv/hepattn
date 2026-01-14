#!/bin/bash

#SBATCH --job-name=trackml-train
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=/home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/trackml/slurm_logs/slurm-%j.%x.out
#SBATCH --time=00:12:00

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Move to workdir
cd /home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/trackml/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

echo "nvidia-smi:"
nvidia-smi

# Run the training
echo "Running training script..."

# Python command that will be run
PYTORCH_CMD="python run_filtering.py fit --config configs/filtering_noComet.yaml --trainer.fast_dev_run 10"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="srun apptainer run --nv --bind /home/xzcapnai/hepattn/hepattn/src/hepattn/experiments/trackml \
						 /home/xzcapnai/hepattn/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
