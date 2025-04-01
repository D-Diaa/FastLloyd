"""
Module for generating heatmap visualizations of clustering algorithm scalability.

This module creates heatmap visualizations to analyze the scalability of different
clustering algorithms across varying numbers of clusters and dimensions. It supports
comparison between different methods (SuLloyd, Lloyd, FastLloyd, GLloyd) and can
generate both absolute performance heatmaps and relative performance comparisons.

The module handles multiple privacy settings (ε values) and can process both
adapted and constant iteration scenarios.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import Bbox
from tqdm import tqdm

quality_key = 'Normalized Intra-cluster Variance (NICV)'


# Refactored code
def extract_data(main_folder, method_names, eps):
    data_dict = {method: {} for method in method_names}

    for folder in tqdm(os.listdir(main_folder)):
        actual_folder = os.path.join(main_folder, folder)
        if folder.startswith("Synth") and os.path.isdir(actual_folder):
            _, k, d, _ = folder.split("_")
            k, d = int(k), int(d)
            if k > 32 or d > 512:
                continue
            for method in method_names:
                dist_method, dp, post = method
                clusters, dimensions = map(int, folder.split('_')[1:3])
                variances_file_path = os.path.join(actual_folder, 'variances.csv')
                try:
                    df = pd.read_csv(variances_file_path)
                    if not isinstance(eps, list):
                        if dp == "none":
                            _eps = 0
                        else:
                            _eps = eps
                        query_str = f"dp == '{dp}' and method == '{dist_method}' and eps == {_eps} and post == '{post}'"
                        row = df.query(query_str)
                        if row.empty or quality_key not in row.columns:
                            print(f"No '{dp}' data for {method} in file {variances_file_path}")
                            continue
                        nicv_value = row[quality_key].iloc[0]

                    else:
                        query_str = f"dp == '{dp}' and method == '{dist_method}' and post == '{post}'"
                        row = df.query(query_str)
                        if row.empty or quality_key not in row.columns:
                            print(f"No '{dp}' data for {method} in file {variances_file_path}")
                            continue
                        nicv_values = row[quality_key].values
                        eps_values = row['eps'].values
                        indices = np.argsort(eps_values)
                        eps_values = eps_values[indices]
                        nicv_values = nicv_values[indices]
                        if len(eps_values) == 1:
                            eps_values = eps
                            nicv_values = np.array([nicv_values[0] for _ in eps_values])
                        assert len(eps_values) == len(nicv_values)
                        nicv_value = np.trapz(nicv_values, x=eps_values)

                    if (clusters, dimensions) not in data_dict[method]:
                        data_dict[method][(clusters, dimensions)] = []
                    data_dict[method][(clusters, dimensions)].append(nicv_value)
                except FileNotFoundError:
                    print(f"File not found: {variances_file_path}")
    for method in method_names:
        for key in data_dict[method]:
            vals = data_dict[method][key]
            data_dict[method][key] = sum(vals) / len(vals)

    return data_dict


def generate_heatmap_from_matrix(matrix, x_labels, y_labels, cmap, vmin, vmax, file_name, fmt=None, no_bar=False):
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 18})

    if fmt is None:
        def custom_format(val):
            if abs(val) < 1:  # Threshold for using scientific notation
                return f"{val:.3f}"  # Scientific notation
            else:
                return f"{val:.3g}"  # 3 significant figures

        annot_matrix = np.vectorize(custom_format)(matrix)

        if no_bar:
            sns.heatmap(matrix, annot=annot_matrix, fmt="", xticklabels=x_labels, yticklabels=y_labels,
                        cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)
        else:
            sns.heatmap(matrix, annot=annot_matrix, fmt="", xticklabels=x_labels, yticklabels=y_labels,
                        cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        if no_bar:
            sns.heatmap(matrix, annot=True, fmt=fmt, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, vmin=vmin,
                        vmax=vmax, cbar=False)
        else:
            sns.heatmap(matrix, annot=True, fmt=fmt, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, vmin=vmin,
                        vmax=vmax)
    plt.xlabel('Number of Dimensions', fontsize=24)
    plt.ylabel('Number of Clusters', fontsize=24)
    plt.xticks(fontsize=22)  # X-ticks font size
    plt.yticks(fontsize=22)  # Y-ticks font size
    plt.gca().invert_yaxis()  # Invert the y-axis
    if not no_bar:
        bbox = Bbox([[7.5, -1], [9, 7.5]])
        plt.savefig(f'{main_folder}/{file_name}.png', bbox_inches=bbox)
    else:
        plt.savefig(f'{main_folder}/{file_name}.png')
    plt.close()


def create_heatmap(data, method_name, vmin, vmax, no_bar=False, eps=1):
    # Extract clusters and dimensions as separate lists
    clusters = sorted(set(key[0] for key in data.keys()))
    dimensions = sorted(set(key[1] for key in data.keys()))

    # Initialize a matrix to store NICV values
    nicv_matrix = np.zeros((len(clusters), len(dimensions)))

    # Populate the matrix with NICV values
    for i, cluster in enumerate(clusters):
        for j, dimension in enumerate(dimensions):
            nicv_matrix[i, j] = data.get((cluster, dimension), np.nan)  # Use NaN for missing values

    # Create a custom colormap: green (low), yellow (middle), red (high)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    name = f'Heatmap_eps{eps}' + ("_bar" if not no_bar else f'_{method_name}')
    generate_heatmap_from_matrix(nicv_matrix, dimensions, clusters, cmap, vmin, vmax, name, no_bar=no_bar)

    return nicv_matrix


method_names = {
    ("none", "laplace", "none"): "SuLloyd",
    ("diagonal_then_frac", "gaussiananalytic", "fold"): "FastLloyd",
}
division_matrices = {
    "Fast vs SU": ("none", "laplace", "none", "diagonal_then_frac", "gaussiananalytic", "fold"),
}

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 1:
        main_folder = sys.argv[1]
    else:
        main_folder = "submission"
    folder = f"{main_folder}/scale"
    print(f"Processing data in {folder}")
    for eps in [[0.1, 0.25, 0.5, 0.75, 1]]:
        try:
            # Extract the data
            data_dict = extract_data(folder, method_names, eps)

            # Find the global min and max NICV values for the shared scale
            all_nicv_values = [value for method_data in data_dict.values() for value in method_data.values()]
            vmin, vmax = min(all_nicv_values), max(all_nicv_values)

            # Generate heatmaps for each methodregation method and store NICV matrices
            nicv_matrices = {}
            for method, data in data_dict.items():
                nicv_matrices[method] = create_heatmap(data, method, vmin, vmax, no_bar=True, eps=eps)

            create_heatmap(data, method, vmin, vmax, no_bar=False, eps=0)
            # Calculate the division matrix (ensure to handle division by zero appropriately)
            for name, keys in division_matrices.items():
                first_matrix = nicv_matrices[keys[:3]]
                second_matrix = nicv_matrices[keys[3:]]
                division_matrix = np.divide(first_matrix - second_matrix, first_matrix, out=np.zeros_like(first_matrix),
                                            where=second_matrix != 0)
                # Assuming division_matrix, and the labels (dimensions and clusters) are already defined
                # cmao with shades of blue
                cmap_division = sns.color_palette("Blues", as_cmap=True)
                vmin_div, vmax_div = np.nanmin(division_matrix), np.nanmax(division_matrix)

                # Generate the heatmap for the division matrix
                generate_heatmap_from_matrix(division_matrix, sorted(set(key[1] for key in data.keys())),
                                             sorted(set(key[0] for key in data.keys())), cmap_division,
                                             vmin_div, vmax_div,
                                             f'Heatmap_eps{eps}_{name}', fmt=".0%", no_bar=True)
        except Exception as e:
            print(f"Failed to process data for {eps}: {e}")
            continue
