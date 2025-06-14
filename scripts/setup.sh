#!/bin/bash
# Setup script for FastLloyd experiments
# This script extracts data and sets up the conda environment

set -e  # Exit on any error

echo "=== FastLloyd Setup Script ==="
echo "Setting up environment and extracting data..."

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Define data directory
DATA_DIR="$PROJECT_ROOT/data"

echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"

# Step 1: Extract all data archives
echo ""
echo "=== Step 1: Extracting Data Archives ==="

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# List of data archives to extract
archives=(
    "ablate_datasets.tar.xz"
    "g2_datasets.tar.xz" 
    "real_datasets.tar.xz"
    "scale_datasets.tar.xz"
    "timing_datasets.tar.xz"
)

# Extract each archive without folder structure (skip if data already exists)
if [ "$(ls -A "$DATA_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Data directory contains files, skipping extraction"
else
    for archive in "${archives[@]}"; do
        if [ -f "$archive" ]; then
            echo "Extracting $archive..."
            tar --extract --xz --file="$archive" --directory="$DATA_DIR"
            echo "✓ $archive extracted"
        else
            echo "⚠ Warning: $archive not found"
        fi
    done
fi

echo "✓ Data extraction completed"
echo "Extracted data to: $DATA_DIR"

# Step 2: Create and install conda environment
echo ""
echo "=== Step 2: Setting up Conda Environment ==="

if [ -f "env.yml" ]; then
    echo "Creating conda environment from env.yml..."
    conda env create -f env.yml
    echo "✓ Conda environment created"
    
    echo "Activating conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate fastlloyd
    echo "✓ Conda environment activated"
else
    echo "⚠ Warning: env.yml not found, skipping environment setup"
fi

echo ""
echo "=== Setup Complete ==="
echo "Environment is ready for experiments!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate fastlloyd"
echo "2. Run experiments: bash scripts/reproduce.sh"