#!/bin/bash
# Run script for FastLloyd experiments
# This script runs all experiments and saves results
# Prerequisites: Run setup.sh first to extract data and create environment

set -e  # Exit on any error

echo "=== FastLloyd Run Script ==="
echo "Running all experiments..."

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Define directories
SUBMISSION_DIR="$PROJECT_ROOT/submission"

echo "Project root: $PROJECT_ROOT"

# Ensure conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "fastlloyd" ]]; then
    echo "⚠ Warning: fastlloyd conda environment not activated"
    echo "Please run: conda activate fastlloyd"
    echo "Or run setup.sh first if environment doesn't exist"
    exit 1
fi

# Create submission directory if it doesn't exist
if [ ! -d "$SUBMISSION_DIR" ]; then
    mkdir -p "$SUBMISSION_DIR"
    echo "Created submission directory: $SUBMISSION_DIR"
fi

echo ""
echo "=== Running Experiments ==="

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Run accuracy and scale experiments
echo "Running accuracy and scale experiments..."
if [ -f "scripts/run_accuracy_scale_experiments.sh" ]; then
    bash scripts/run_accuracy_scale_experiments.sh
    echo "✓ Accuracy and scale experiments completed"
    echo "Experiment results saved to: $SUBMISSION_DIR"
else
    echo "⚠ Warning: run_accuracy_scale_experiments.sh not found"
fi

# Run timing experiments
echo ""
echo "Running timing experiments..."
if [ -f "scripts/run_timing_experiments.sh" ]; then
    bash scripts/run_timing_experiments.sh
    echo "✓ Timing experiments completed"
    echo "Timing results saved to: $SUBMISSION_DIR"
else
    echo "⚠ Warning: run_timing_experiments.sh not found"
fi

echo ""
echo "=== Experiments Complete ==="
echo "All experiment results saved to: $SUBMISSION_DIR"
echo ""
echo "Next step: Run 'bash scripts/generate_plots.sh' to generate plots and analysis"