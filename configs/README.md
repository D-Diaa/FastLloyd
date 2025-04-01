# Federated Clustering Configs

A Python package for managing configuration parameters in privacy-preserving federated clustering experiments. This package provides a flexible and robust way to handle experimental settings, privacy mechanisms, and clustering parameters.

## Overview

The `configs` package consists of three main components:

1. `Params`: A configuration class that manages clustering and privacy parameters, including:
   - Clustering parameters (dimensions, number of clusters, iterations)
   - Privacy mechanism settings (epsilon budgets, noise mechanisms)
   - Federated learning parameters (number of clients, communication delays)
   - Dynamic constraint handling for centroid updates

2. `defaults`: Pre-configured experimental settings for:
   - Timing experiments (computational performance)
   - Accuracy experiments (clustering quality)
   - Scaling experiments (dataset size variations)
   - Ablation studies (parameter sensitivity analysis)

3. Dataset configurations for various experimental scenarios, including:
   - Real-world benchmark datasets
   - Synthetic datasets with controlled parameters

## Usage

```python
from configs import Params, exp_parameter_dict, num_clusters

# Create a configuration with custom parameters
params = Params(
    dim=2,
    k=15,
    eps=0.1,
    dp="gaussiananalytic"
)

# Access pre-configured experimental settings
timing_config = exp_parameter_dict["timing"]
accuracy_config = exp_parameter_dict["accuracy"]

# Get number of clusters for a specific dataset
n_clusters = num_clusters["iris"]  # Returns 3
```

## Features

- **Privacy Mechanisms**: Supports Laplace and Gaussian Analytic differential privacy
- **Dynamic Parameters**: Automatic adjustment of iterations and privacy budgets
- **Constraint Methods**: Various methods for bounding centroid updates
- **Reproducibility**: Consistent random seed handling for experiments
- **Flexible Configuration**: Easy parameter overrides via keyword arguments