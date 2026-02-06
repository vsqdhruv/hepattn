#!/bin/bash

# Script to open jupyter notebook #

echo "Navigating into Mu3e directory..."
cd src/hepattn/experiments/mu3e

echo "Activating virtual environment..."
source venv/bin/activate

echo "Opening JupyterLab..."
jupyter lab

echo ""
echo "JupyterLab opened!"