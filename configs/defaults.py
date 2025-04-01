"""Default configuration parameters for federated clustering experiments.

This module defines the default parameters and dataset configurations for various
experimental setups in privacy-preserving federated clustering. It includes settings
for different types of experiments:
- Timing experiments (measuring computational performance)
- Accuracy experiments (measuring clustering quality)
- Scaling experiments (testing with different dataset sizes)
- Ablation studies (analyzing impact of specific parameters)

The configurations cover:
- Dataset selections
- Privacy mechanisms and parameters
- Experimental variables
- Cluster counts for different datasets
"""

import os

# Dataset configurations for ablation studies
ablate_dataset = [
    f"AblateSynth_{k}_{d}_{sep}"
    for k in [2, 4, 8, 16]  # Number of clusters
    for d in [2, 4, 8, 16]  # Dimensions
    for sep in [0.25, 0.5, 0.75]  # Cluster separation
]

# G2 datasets (gathered from data directory)
g2 = [file.replace(".txt", "") for file in os.listdir("data") if file.startswith("g2")]

# Real-world benchmark datasets
real_datasets = ["iris", "s1", "house", "adult", "lsun", "birch2", "wine", "yeast", "breast", "mnist"]

# Combined datasets for accuracy experiments
accuracy_datasets = g2 + real_datasets

# Datasets for timing experiments
timing_datasets = [
                      "s1", "lsun"  # Real datasets
                  ] + [
                      # Synthetic datasets with varying parameters
                      f"timesynth_{k}_{d}_{n}"
                      for k in [2, 5]  # Number of clusters
                      for d in [2, 5]  # Dimensions
                      for n in [10000, 100000]  # Number of points
                  ]

# Datasets for scaling experiments
scale_datasets = [file.replace(".txt", "") for file in os.listdir("data") if file.startswith("Synth")]

# Experimental configurations

timing_parameters = {
    "dps": ["gaussiananalytic"],  # Privacy mechanisms
    "eps_budgets": [0.1],  # Privacy budgets
    "delays": [0.000125, 0.025],  # Communication delays
    "datasets": timing_datasets,  # Datasets to use
}

acc_parameters = {
    "dps": ["none", "laplace", "gaussiananalytic"],  # Privacy mechanisms
    "methods": ["none", "diagonal_then_frac"],  # Constraint methods
    "eps_budgets": [0.1, 0.25, 0.5, 0.75, 1],  # Privacy budgets
    "posts": ["none", "fold"],  # Post-processing methods
    "datasets": accuracy_datasets,  # Datasets to use
}

scale_parameters = {
    "dps": ["none", "laplace", "gaussiananalytic"],  # Privacy mechanisms
    "methods": ["none", "diagonal_then_frac"],  # Constraint methods
    "eps_budgets": [0.1, 0.25, 0.5, 0.75, 1],  # Privacy budgets
    "posts": ["none", "fold"],  # Post-processing methods
    "datasets": scale_datasets,  # Datasets to use
}

ablation_parameters = {
    "dps": ["gaussiananalytic"],  # Privacy mechanisms
    "methods": ["none", "diagonal_then_frac", "frac_stay"],
    "eps_budgets": [0.1],
    "posts": ["none", "truncate", "fold", "none_unclipped", "truncate_unclipped", "fold_unclipped"],
    "datasets": ablate_dataset,  # Datasets to use
    "alphas": [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,
               0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.25,
               1.3, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 1.9, 2.0],
}

# Number of clusters for each dataset
num_clusters = {
    # Real-world datasets
    "iris": 3,
    "s1": 15,
    "birch2": 100,
    "house": 3,
    "adult": 3,
    "lsun": 3,
    "wine": 3,
    "yeast": 10,
    "breast": 2,
    "mnist": 10,
}

num_clusters.update({
    dataset: 2 for dataset in g2  # G2 datasets
})

# Dictionary mapping experiment types to their parameters
exp_parameter_dict = {
    "timing": timing_parameters,  # Timing experiments
    "accuracy": acc_parameters,  # Accuracy experiments
    "ablation": ablation_parameters,  # Maximum distance ablation study
    "scale": scale_parameters,  # Scaling experiments
}
