#!/bin/bash
# Full end-to-end pipeline for FastLloyd experiments
# This script runs setup, experiments, and generates all plots/analysis
# No prerequisites - runs everything from scratch

set -e  # Exit on any error

echo "=== FastLloyd Full Pipeline ==="
echo "Running complete end-to-end pipeline (setup + experiments + plots)..."

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# Step 1: Setup environment and data
echo ""
echo "=== Step 1: Environment Setup ==="
if [ -f "scripts/setup.sh" ]; then
    bash scripts/setup.sh
    echo "✓ Setup completed"
else
    echo "⚠ Error: setup.sh script not found"
    exit 1
fi

# Step 2: Run all experiments
echo ""
echo "=== Step 2: Running All Experiments ==="
# Ensure conda environment is activated
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastlloyd

if [ -f "scripts/run_experiments.sh" ]; then
    bash scripts/run_experiments.sh
    echo "✓ All experiments completed"
else
    echo "⚠ Error: run_experiments.sh script not found"
    exit 1
fi

# Step 3: Generate all plots and analysis
echo ""
echo "=== Step 3: Generating All Plots and Analysis ==="
if [ -f "scripts/generate_plots.sh" ]; then
    bash scripts/generate_plots.sh
    echo "✓ All plots and analysis generated"
else
    echo "⚠ Error: generate_plots.sh script not found"
    exit 1
fi

# Summary
echo ""
echo "=== Full Pipeline Complete ==="
echo "Complete FastLloyd reproduction finished successfully!"
echo ""
echo "What was accomplished:"
echo "• Environment setup and data extraction"
echo "• All experiments executed"
echo "• All plots and analysis generated"
echo ""
echo "Individual script usage:"
echo "• Setup only: bash scripts/setup.sh"
echo "• Run experiments only: bash scripts/run_experiments.sh"
echo "• Generate plots only: bash scripts/generate_plots.sh"
echo "• Experiments + plots: bash scripts/experiments_and_plots.sh"
echo "• Complete reproduction: bash scripts/complete_reproduction.sh (this script)"