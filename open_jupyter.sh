#!/bin/bash

# Script to open jupyter notebook #co
conda deactivate

echo "Activating virtual environment..."
source ~/mu3e_env/bin/activate

echo "Loading Python 3.9.6.."
module load Python/3.9.6-GCCcore-11.2.0

echo "Opening JupyterLab..."
jupyter lab

echo ""
echo "JupyterLab opened!"