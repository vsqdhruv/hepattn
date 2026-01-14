#!/bin/bash

# Setup script for ROOT data analysis environment

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing ROOT, Jupyter, and dependencies..."
pip install jupyterlab notebook uproot awkward matplotlib numpy pandas

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter Lab, run:"
echo "  jupyter lab"
echo ""
echo "An initial notebook 'explore_mu3e.ipynb' has been created for you."