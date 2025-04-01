"""
Module for creating ablation study plots for clustering algorithms.

This module provides functionality to analyze and visualize the performance of
different clustering methods across various parameters and privacy settings.
It focuses on plotting Normalized Intra-cluster Variance (NICV) against different
parameters while considering different privacy mechanisms and dataset characteristics.
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from configs import exp_parameter_dict

# Constants
QUALITY_KEY = 'Normalized Intra-cluster Variance (NICV)'
FIGURE_SIZE = (15, 15)
COLOR_MAP_NAME = 'tab10'
plt.rcParams.update({'font.size': 20})


def extract_meta(dataset_name: str) -> Tuple[int, int]:
    if "synth" in dataset_name.lower():
        _, k, d, _ = dataset_name.split("_")
        return int(k), int(d)
    raise ValueError(f"Unknown dataset format: {dataset_name}")


def create_style_mappings(methods: list) -> Tuple[Dict, Dict, Dict]:
    color_map = plt.colormaps[COLOR_MAP_NAME]
    unique_methods = set(method[0] for method in methods)
    color_mapping = {method: color_map(i) for i, method in enumerate(unique_methods)}
    marker_mapping = {"unclipped": 'X', "clipped": 'o'}
    line_mapping = {
        "none": '-',
        "truncate": '--',
        "fold": '-.',
    }
    return color_mapping, marker_mapping, line_mapping


def normalize_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    norm_mask = (data['dp'] == "gaussiananalytic") & (data['method'] == "none") & (data['post'] == "none")
    norm_value = data[norm_mask][QUALITY_KEY].values[0]
    return data, norm_value


def calculate_averages(xs: Dict[str, List], ys: Dict[str, List], all_xs: np.ndarray,
                       pre: str, suff: str) -> List[float]:
    avg_ys = []
    for x in all_xs:
        y_values = [ys[_dataset][i]
                    for _dataset in xs
                    if x in xs[_dataset]
                    for i, x_val in enumerate(xs[_dataset])
                    if x_val == x and _dataset.startswith(pre) and _dataset.endswith(suff)]
        avg_ys.append(np.mean(y_values) if y_values else np.nan)
    return avg_ys


dist_method_labels = {
    "none": "No Constraint",
    "diagonal_then_frac": "Step",
    "frac_stay": "Constant"
}
post_method_labels = {
    "none": "No Post-Processing",
    "truncate": "Truncate",
    "fold": "Fold",
}
clip_method_labels = {
    "clipped": "Radius-Clipped",
    "unclipped": "No Clipping"
}


def create_legend_handles(color_mapping: Dict, marker_mapping: Dict, line_mapping: Dict) -> List[Line2D]:
    color_handles = [Line2D([0], [0], color=color, lw=3, label=dist_method_labels[method])
                     for method, color in color_mapping.items()]
    line_handles = [Line2D([0], [0], color='black', lw=3, linestyle=linestyle, label=post_method_labels[p]) for
                    p, linestyle in line_mapping.items()]
    marker_handles = [Line2D([0], [0], color='black', marker=marker, lw=1, markersize=10,
                             label=clip_method_labels[p])
                      for p, marker in marker_mapping.items()]

    return color_handles + marker_handles + line_handles


def plot_nicv_vs_key(data: Dict[str, pd.DataFrame], key: str, folder: str, eps: float) -> None:
    plt.figure(figsize=FIGURE_SIZE)

    # Extract unique parameter values and create style mappings
    sample = data[list(data.keys())[0]]
    methods = sample[['method', 'post']].drop_duplicates().values.tolist()
    methods = [tuple(method) for method in methods]

    color_mapping, marker_mapping, line_mapping = create_style_mappings(methods)

    # Initialize data structures
    xs = {}
    ys = {}
    grouped_data = defaultdict(list)

    # Process each dataset
    for dataset in data:
        k, d = extract_meta(dataset)
        dataset_data, norm_value = normalize_data(data[dataset])

        for method in methods:
            exp_key = f"{k}_{d}_{method}"
            mask = ((dataset_data['method'] == method[0]) &
                    (dataset_data['post'] == method[1]))
            exp_data = dataset_data[mask]
            sorted_data = exp_data.sort_values(key)

            if len(sorted_data) == 0:
                continue

            xs[exp_key] = sorted_data[key].values.tolist()
            ys[exp_key] = (sorted_data[QUALITY_KEY].values / norm_value).tolist()
            grouped_data[exp_key].append(ys[exp_key])

    # Plot the data
    all_xs = np.unique(np.concatenate(list(xs.values())))

    # Plot averages
    for method in methods:
        linestring = method[1].replace("_unclipped", "")
        linestyle = line_mapping[linestring]
        markerstring = "unclipped" if "unclipped" in method[1] else "clipped"
        marker = marker_mapping[markerstring]
        color = color_mapping[method[0]]
        # Overall average
        avg_ys = calculate_averages(xs, ys, all_xs, "", f"_{method}")
        if np.isnan(avg_ys[1]):
            if method[0] == "none" and "unclipped" not in method[1]:
                plt.axhline(y=avg_ys[0], linewidth=2, color=color, linestyle=linestyle)
        else:
            plt.plot(all_xs, avg_ys, linewidth=2, color=color, linestyle=linestyle, marker=marker, markersize=10)

    # Customize plot
    plt.xlabel(key)
    plt.ylabel('NICV')
    plt.grid(True)
    plt.xlim([0.4, 1.8])
    plt.ylim([0.6, 1.5])
    # Add legen
    legend_handles = create_legend_handles(color_mapping, marker_mapping, line_mapping)
    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

    plt.tight_layout()

    # Save plot
    save_folder = f"figs/{folder}"
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{save_folder}/ablation_{eps}.pdf")
    plt.close()


def process_experiment_data(folder: str, param_name: str) -> None:
    per_eps = defaultdict(dict)

    # Read and organize data
    for dataset in exp_parameter_dict[f'ablation']['datasets']:
        file_path = f'{folder}/ablation/{dataset}/variances.csv'
        if not os.path.isfile(file_path):
            continue

        print(f"Processing: {file_path}")
        data = pd.read_csv(file_path)

        # Partition data by epsilon values
        for eps, eps_data in data.groupby('eps'):
            per_eps[eps][dataset] = eps_data

    # Generate plots for each epsilon value
    for eps, eps_data in per_eps.items():
        plot_nicv_vs_key(eps_data, key=param_name, folder=folder, eps=eps)


def main(folder):
    print(f"\nProcessing folder: {folder}")
    try:
        process_experiment_data(folder, "alpha")
        print(f"Successfully processed {folder}")
    except Exception as e:
        print(f"Error processing {folder}: {str(e)}")


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 1:
        main_folder = sys.argv[1]
    else:
        main_folder = "submission"
    main(main_folder)
