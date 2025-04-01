"""
Module for creating bar plots comparing clustering algorithm performance on synthetic datasets.

This module processes and visualizes the performance of different clustering methods
on synthetic datasets, focusing on comparing their Area Under Curve (AUC) metrics
across different dimensions and dataset families. It supports multiple dataset
families (e.g., 'g2', 'dim') and normalizes results relative to SuLloyd's performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

plt.rcParams.update({'font.size': 32})
# Define a standard range of epsilon values for interpolation
standard_epsilons = [0.1, 0.25, 0.5, 0.75, 1]

# Define the mapping for method names based on "method" and "dp" columns
method_names = {
    ("none", "laplace", "none"): "SuLloyd",
    ("none", "none", "none"): "Lloyd",
    ("none", "gaussiananalytic", "none"): "GLloyd",
    ("diagonal_then_frac", "gaussiananalytic", "fold"): "FastLloyd",
}
method_colors = {
    "SuLloyd": "red",
    "Lloyd": "black",
    "GLloyd": "orange",
    "FastLloyd": "green",
}


def load_data(data_dir, dataset_prefix, dataset_suffix=""):
    """Load and preprocess data from the specified directory and dataset family.

    Args:
        data_dir (str): Path to the data directory.
        dataset_prefix (str): Dataset family prefix (e.g., 'g2', 'dim').
        dataset_suffix (str): Optional suffix for filtering dataset folders.

    Returns:
        dict: A dictionary with dimensions as keys and AUC values per method as values.
    """
    repeat_suffixes = ["_1", "_2", "_3"] if "Synth" in dataset_prefix else [""]
    repeat_suffixes = [dataset_suffix + suffix for suffix in repeat_suffixes]
    dataset_folders = [folder for folder in os.listdir(data_dir) if
                       folder.startswith(dataset_prefix) and any(folder.endswith(suffix) for suffix in repeat_suffixes)]

    # Initialize dictionary to store aggregated data
    dimension_data_auc = {}
    cluster_data_auc = {}

    for folder in dataset_folders:
        folder_path = os.path.join(data_dir, folder)
        csv_path = os.path.join(folder_path, 'variances.csv')

        # Check if variances.csv exists
        if os.path.exists(csv_path):
            # Read the CSV file
            data = pd.read_csv(csv_path)

            # Extract the dimension from the folder name
            if dataset_prefix == 'g2':
                dimension = int(folder.split('-')[1])
                cluster = 2
            elif dataset_prefix == 'dim':
                dimension = int(folder.replace('dim', ''))
                if dimension < 15:
                    cluster = 9
                else:
                    cluster = 16
            elif dataset_prefix.startswith('Synth'):
                dimension = int(folder.split('_')[2])
                cluster = int(folder.split('_')[1])
            else:
                raise ValueError(f"Unknown dataset family: {dataset_prefix}")

            # Aggregate NICV values by method and dimension
            if dimension not in dimension_data_auc:
                dimension_data_auc[dimension] = {}

            if cluster not in cluster_data_auc:
                cluster_data_auc[cluster] = {}

            for (method, dp, post), group in data.groupby(['method', 'dp', 'post']):
                if (method, dp, post) in method_names:
                    method_name = method_names[(method, dp, post)]
                    epsilons = group['eps'].values
                    sorted_indices = np.argsort(epsilons)
                    epsilons = epsilons[sorted_indices]
                    nicvs = group['Normalized Intra-cluster Variance (NICV)'].values
                    nicvs = nicvs[sorted_indices]

                    # Handle cases with only one point
                    if len(epsilons) == 1:
                        epsilons = standard_epsilons
                        nicvs = np.full_like(standard_epsilons, nicvs[0])

                    assert len(epsilons) == len(nicvs) == len(standard_epsilons)
                    # Calculate the AUC for the method and dimension
                    auc = np.trapz(nicvs, x=standard_epsilons)
                    if method_name not in dimension_data_auc[dimension]:
                        dimension_data_auc[dimension][method_name] = 0
                    if method_name not in cluster_data_auc[cluster]:
                        cluster_data_auc[cluster][method_name] = 0
                    # For dimensions with multiple datasets, sum the AUC values
                    dimension_data_auc[dimension][method_name] += auc
                    cluster_data_auc[cluster][method_name] += auc

    # Normalize AUC values by SuLloyd
    for dimension, methods in dimension_data_auc.items():
        sulloyd_auc = methods.get("SuLloyd", 1)  # Avoid division by zero
        for method in methods:
            methods[method] /= sulloyd_auc

    for cluster, methods in cluster_data_auc.items():
        sulloyd_auc = methods.get("SuLloyd", 1)
        for method in methods:
            methods[method] /= sulloyd_auc

    return dimension_data_auc, cluster_data_auc


def plot_data(dimension_data_auc, dataset, data_dir, type="dimension"):
    """Plot the normalized AUC data.

    Args:
        dimension_data_auc (dict): Aggregated data for plotting.
        dataset (str): Dataset name for labeling.
        data_dir (str): Directory to save the plots.
        type (str): Type of plot to generate (default: "dimension").
    """
    # Convert to DataFrame for easier plotting
    df_auc = pd.DataFrame(dimension_data_auc).T

    # Sort by dimension
    df_auc.sort_index(inplace=True)

    # Plotting
    plt.figure(figsize=(16, 8))
    methods_auc = df_auc.columns
    x = np.arange(len(df_auc.index))  # Number of dimensions

    # Plot bars for each method
    bar_width = 0.2
    for i, method in enumerate(methods_auc):
        plt.bar(x + i * bar_width, df_auc[method], width=bar_width, label=method, color=method_colors[method])

    # Formatting
    if type == "dimension":
        plt.xlabel("Number of Dimensions")
        plt.ylabel("Normalized AUC")
        # plt.title(f"Normalized AUC of NICV per Method for Different Dimensions ({dataset} Datasets)")
    elif type == "cluster":
        plt.xlabel("Number of Clusters")
        plt.ylabel("Normalized AUC")
        # plt.title(f"Normalized AUC of NICV per Method for Different Clusters ({dataset} Datasets)")
    plt.xticks(x + bar_width * (len(methods_auc) - 1) / 2, df_auc.index)
    # plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.yscale('log')

    # Save the plot
    output_path = os.path.join(data_dir, f"{dataset}_{type}_auc.pdf")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    """Main function to execute the data loading, processing, and plotting."""
    parser = argparse.ArgumentParser(description="Process NICV data and plot AUC.")
    parser.add_argument("--data_dir", type=str, default="submission",
                        help="Path to the directory containing dataset folders.")
    args = parser.parse_args()
    prefixes = ["g2"]
    suffixes = [""]
    data_dir = os.path.join(args.data_dir, "accuracy")
    for dataset_prefix, dataset_suffix in zip(prefixes, suffixes):
        dimension_data_auc, cluster_data_auc = load_data(data_dir, dataset_prefix, dataset_suffix)
        dataset = dataset_prefix + dataset_suffix
        plot_data(dimension_data_auc, dataset, args.data_dir, type="dimension")
        plot_data(cluster_data_auc, dataset, args.data_dir, type="cluster")


if __name__ == "__main__":
    main()
