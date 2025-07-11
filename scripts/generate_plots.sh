#!/bin/bash
# Report script for FastLloyd experiments
# This script generates all plots and analysis from experiment results
# Prerequisites: Run setup.sh and run_experiments.sh first

set -e  # Exit on any error

echo "=== FastLloyd Report Script ==="
echo "Generating plots and analysis from experiment results..."

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

# Check if submission directory exists (should contain experiment results)
if [ ! -d "$SUBMISSION_DIR" ]; then
    echo "⚠ Error: Submission directory not found: $SUBMISSION_DIR"
    echo "Please run 'bash scripts/run_experiments.sh' first to generate experiment results"
    exit 1
fi

echo ""
echo "=== Generating Plots and Analysis ==="

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Per-dataset plots
echo "Generating per-dataset plots..."
if [ -f "plots/per_dataset.py" ]; then
    python plots/per_dataset.py "$SUBMISSION_DIR"
    echo "✓ Per-dataset plots generated"
    echo "Per-dataset plots saved to: $SUBMISSION_DIR/accuracy/[dataset_name]/[dataset_name]_[metric].png"
else
    echo "⚠ Warning: per_dataset.py not found"
fi

# Scale heatmap plots
echo ""
echo "Generating scale heatmap plots..."
if [ -f "plots/scale_heatmap.py" ]; then
    python plots/scale_heatmap.py "$SUBMISSION_DIR"
    echo "✓ Scale heatmap plots generated"
    echo "Scale heatmap plots saved to: $SUBMISSION_DIR/scale/Heatmap_*.png"
else
    echo "⚠ Warning: scale_heatmap.py not found"
fi

# Synthetic bar plots for g2 datasets
echo ""
echo "Generating synthetic bar plots for g2 datasets..."
if [ -f "plots/synthetic_bar.py" ]; then
    python plots/synthetic_bar.py --data_dir "$SUBMISSION_DIR" --dataset g2
    echo "✓ G2 synthetic bar plots generated"
    echo "G2 bar plot saved to: $SUBMISSION_DIR/g2_dimension_auc.pdf"
else
    echo "⚠ Warning: synthetic_bar.py not found"
fi

# Synthetic bar plots for Synth datasets
echo ""
echo "Generating synthetic bar plots for Synth datasets..."
if [ -f "plots/synthetic_bar.py" ]; then
    python plots/synthetic_bar.py --data_dir "$SUBMISSION_DIR" --dataset Synth
    echo "✓ Synth synthetic bar plots generated"
    echo "Synth bar plot saved to: $SUBMISSION_DIR/Synth_2_cluster_auc.pdf"
else
    echo "⚠ Warning: synthetic_bar.py not found"
fi

# Timing analysis plots
echo ""
echo "Generating timing analysis plots..."
if [ -f "plots/timing_analysis.py" ]; then
    python plots/timing_analysis.py "$SUBMISSION_DIR"
    echo "✓ Timing analysis plots generated"
    echo "Timing analysis results saved to: $SUBMISSION_DIR/timing_[n]/table.csv and $SUBMISSION_DIR/table_[n].tex"
else
    echo "⚠ Warning: timing_analysis.py not found"
fi

echo ""
echo "=== Report Generation Complete ==="
echo "All plots and analysis have been generated!"
echo ""
echo "Output locations:"
echo "• Per-dataset plots: $SUBMISSION_DIR/accuracy/[dataset_name]/"
echo "• Scale heatmaps: $SUBMISSION_DIR/scale/"
echo "• G2 bar chart: $SUBMISSION_DIR/g2_dimension_auc.pdf"
echo "• Synth bar chart: $SUBMISSION_DIR/Synth_2_cluster_auc.pdf"
echo "• Timing analysis: $SUBMISSION_DIR/timing_[n]/ and $SUBMISSION_DIR/table_[n].tex"
echo ""
echo "Use 'ls -la $SUBMISSION_DIR' to see all generated files."