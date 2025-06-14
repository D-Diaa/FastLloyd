#!/bin/bash
# Reproduction script for FastLloyd experiments
# This script runs all experiments and generates all plots
# Prerequisites: Run setup.sh first to extract data and create environment

set -e  # Exit on any error

echo "=== FastLloyd Reproduction Script ==="
echo "Running full reproduction (experiments + plots)..."

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# Ensure conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "fastlloyd" ]]; then
    echo "⚠ Warning: fastlloyd conda environment not activated"
    echo "Please run: conda activate fastlloyd"
    echo "Or run setup.sh first if environment doesn't exist"
    exit 1
fi

# Step 1: Run all experiments
echo ""
echo "=== Step 1: Running All Experiments ==="
if [ -f "scripts/run_experiments.sh" ]; then
    bash scripts/run_experiments.sh
    echo "✓ All experiments completed"
else
    echo "⚠ Error: run_experiments.sh script not found"
    exit 1
fi

# Step 2: Generate all plots and analysis
echo ""
echo "=== Step 2: Generating All Plots and Analysis ==="
if [ -f "scripts/generate_plots.sh" ]; then
    bash scripts/generate_plots.sh
    echo "✓ All plots and analysis generated"
else
    echo "⚠ Error: generate_plots.sh script not found"
    exit 1
fi

# Summary
echo ""
echo "=== Full Reproduction Complete ==="
echo "All experiments, plots, and analysis have been generated!"
echo ""
echo "Individual script usage:"
echo "• Run experiments only: bash scripts/run_experiments.sh"
echo "• Generate plots only: bash scripts/generate_plots.sh"
echo "• Experiments + plots: bash scripts/experiments_and_plots.sh (this script)"