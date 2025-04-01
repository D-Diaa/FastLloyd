# Clustering Analysis and Visualization Tools

This package provides comprehensive visualization and analysis tools for evaluating privacy-preserving federated clustering algorithms. It includes specialized modules for generating various types of plots and analyses, helping researchers understand algorithm performance across multiple dimensions.

## Overview

The package consists of five main modules, each focused on a specific aspect of clustering algorithm analysis:

### 1. Ablation Studies (`ablation_plots.py`)

This module creates visualizations for understanding how different parameters affect clustering performance. It specializes in analyzing the relationship between clustering quality metrics (particularly Normalized Intra-cluster Variance) and various algorithm parameters.

```python
from plots.ablation_plots import process_experiment_data

# Generate ablation plots for a specific parameter
process_experiment_data("results_folder", "alpha")
```

Features:
- Plots NICV against different parameter values
- Supports multiple privacy mechanisms and constraint methods
- Handles various dataset dimensions and cluster counts
- Creates publication-quality plots with clear styling
- Automatically processes multiple privacy budget (ε) values

### 2. Per-Dataset Analysis (`per_dataset.py`)

This module generates detailed performance comparisons across different clustering methods for individual datasets. It compares various algorithms (SuLloyd, GLloyd, FastLloyd, and Lloyd) across different privacy parameters.

```python
from plots.per_dataset import process_datasets, CONFIG

# Configure visualization settings
config = CONFIG.copy()
config['datasets_folders'] = ["your_results/accuracy"]

# Generate per-dataset visualizations
process_datasets(config)
```

Key capabilities:
- Creates comparative plots for multiple clustering metrics
- Supports both constant and adapted iteration scenarios
- Generates consistent legends across all plots
- Handles confidence intervals for robust analysis
- Automatically processes multiple datasets

### 3. Scalability Analysis (`scale_heatmap.py`)

This module creates heatmap visualizations to analyze how clustering algorithms scale with varying numbers of clusters and dimensions. It provides both absolute performance visualization and relative performance comparisons.

```python
from plots.scale_heatmap import extract_data, create_heatmap

# Extract scalability data
data_dict = extract_data(
    "results_folder",
    method_names={("none", "laplace", "none"): "SuLloyd"},
    eps=0.1
)

# Create scalability heatmap
create_heatmap(data_dict["SuLloyd"], "SuLloyd", vmin, vmax)
```

Features:
- Generates heatmaps showing performance across different scales
- Supports comparison between different methods
- Handles multiple privacy budget (ε) values
- Creates both absolute and relative performance visualizations
- Automatically processes different dataset sizes and dimensions

### 4. Synthetic Data Analysis (`synthetic_bar.py`)

This module creates bar plots comparing algorithm performance on synthetic datasets, focusing on Area Under Curve (AUC) metrics across different dimensions and dataset families.

```python
from plots.synthetic_bar import load_data, plot_data

# Load and process synthetic dataset results
dimension_data, cluster_data = load_data(
    "results_dir",
    dataset_prefix="Synth_",
    dataset_suffix="_2"
)

# Create comparison plots
plot_data(dimension_data, "Synthetic", "results_dir", type="dimension")
plot_data(cluster_data, "Synthetic", "results_dir", type="cluster")
```

Features:
- Normalizes results relative to SuLloyd's performance
- Supports multiple dataset families
- Creates dimension-wise and cluster-wise comparisons
- Generates publication-ready bar plots
- Handles multiple experimental repetitions

### 5. Timing Analysis (`timing_analysis.py`)

This module analyzes execution timing data under different network conditions (LAN and WAN), generating comprehensive timing reports.

```python
from plots.timing_analysis import process_all_datasets

# Process timing data and generate reports
results_df = process_all_datasets("timing_results_dir")

# Export results in multiple formats
results_df.to_csv("timing_table.csv")
results_df.to_latex("timing_table.tex")
```

Features:
- Processes timing data for both LAN and WAN scenarios
- Calculates confidence intervals for timing measurements
- Generates formatted output in CSV and LaTeX
- Analyzes communication overhead
- Supports varying numbers of clients

## Usage Example

Here's a complete example showing how to generate a comprehensive analysis:

```python
from plots import (
    process_experiment_data,
    process_datasets,
    create_heatmap,
    plot_data,
    process_all_datasets
)

# Configure analysis parameters
base_dir = "experiment_results"
config = {
    'eps_range': [0, 1],
    'datasets_folders': [f"{base_dir}/accuracy"]
}

# Generate ablation plots
process_experiment_data(f"{base_dir}/ablation", "alpha")

# Create per-dataset visualizations
process_datasets(config)

# Generate scalability heatmaps
data_dict = extract_data(f"{base_dir}/scale", method_names, eps=0.1)
create_heatmap(data_dict, "method_comparison", vmin, vmax)

# Create synthetic data comparisons
dimension_data, cluster_data = load_data(f"{base_dir}/synthetic")
plot_data(dimension_data, "synthetic_comparison", base_dir)

# Generate timing analysis
timing_results = process_all_datasets(f"{base_dir}/timing")
timing_results.to_latex("timing_results.tex")
```

This package is designed to support thorough analysis of clustering algorithm performance, with particular attention to privacy-preserving aspects and scalability characteristics. Each module can be used independently or as part of a comprehensive analysis pipeline.