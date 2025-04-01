"""
Module for generating per-dataset performance visualizations of clustering algorithms.

This module creates detailed plots comparing different clustering methods (SuLloyd,
GLloyd, FastLloyd, and Lloyd) across various datasets. It visualizes metrics such
as Normalized Intra-cluster Variance (NICV) and Empty Clusters count against privacy
parameters (ε).

The module supports customizable plotting configurations and handles both
constant and adapted iteration scenarios.
"""

import os
import sys
from os.path import isdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm

plt.rcParams.update({'font.size': 16})


# Configuration for legend handles and styling
def create_legend_handles():
    method_styles = [
        {'marker': 'o', 'color': 'red', 'label': 'SuLloyd'},
        {'marker': 'o', 'color': 'green', 'label': 'FastLloyd'},
        {'marker': 'o', 'color': 'orange', 'label': 'GLloyd'},
        {'marker': 'o', 'color': 'black', 'label': 'Lloyd'}
    ]
    method_handles = [Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=10,
                             label=style['label']) for style in method_styles]

    return method_handles


def create_legend_image(config):
    # Initialize a figure with a specific size that might need adjustment
    fig, ax = plt.subplots(figsize=(6, 1))  # Adjust figsize to fit the legend as needed
    # Generate the legend handles using the previously defined function
    legend_handles = create_legend_handles()
    # Number of columns set to the number of legend handles to align them horizontally
    ncol = len(legend_handles)
    # Create the legend with the handles, specifying the number of columns
    ax.legend(handles=legend_handles, loc='center', ncol=ncol, frameon=False)
    # Hide the axes as they are not needed for the legend
    ax.axis('off')
    # Remove all the margins and paddings by setting bbox_inches to 'tight' and pad_inches to 0
    # The dpi (dots per inch) parameter might be adjusted for higher resolution
    fig.savefig(os.path.join(config['datasets_folders'][0], "legend.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # Clear the plot to free up memory
    plt.clf()


# Constants and configurations
CONFIG = {
    'eps_range': [0, 1],
    'method_names': {
        ("none", "laplace", "none"): "SuLloyd",
        ("none", "gaussiananalytic", "none"): "GLloyd",
        ("none", "none", "none"): "Lloyd",
        ("diagonal_then_frac", "gaussiananalytic", "fold"): "FastLloyd",

    },
    "method_colors": {
        ("none", "laplace", "none"): "red",
        ("none", "none", "none"): "black",
        ("none", "gaussiananalytic", "none"): "orange",
        ("diagonal_then_frac", "gaussiananalytic", "fold"): "green",
    },
    'datasets_folders': [
        "submission/accuracy"
    ],
    'metrics': ["Normalized Intra-cluster Variance (NICV)"]
}
metrics_dict = {
    "Normalized Intra-cluster Variance (NICV)": "NICV",
}


# Function to process datasets and generate plots
def process_datasets(config):
    for dataset_folder in config['datasets_folders']:
        datasets = list(os.listdir(dataset_folder))
        for dataset in tqdm(datasets):
            folder = os.path.join(dataset_folder, dataset)
            if not isdir(folder):
                continue
            filepath = os.path.join(folder, "variances.csv")
            if not os.path.exists(filepath):
                print(f"FAILED: {folder}")
                continue

            sample_data = pd.read_csv(filepath)
            config['dataset'] = dataset
            plot_data(sample_data, folder, config)


def plot_data(data, folder, config):
    for metric in config['metrics']:
        filtered_data = data[['method', 'dp', 'eps', 'post', metric, f"{metric}_h"]].sort_values(by='eps')
        combinations = filtered_data[['method', 'dp', 'post']].drop_duplicates().values
        for method, dp, post in combinations:
            plot_metric(filtered_data, method, dp, post, metric, config)

        finalize_plot(metric, folder, config["dataset"])


def plot_metric(data, method, dp, post, metric, config):
    # rename metric to be more descriptive
    subset = data[(data['method'] == method) & (data['dp'] == dp) & (data['post'] == post)]
    method_name = (method, dp, post)
    if method_name not in config['method_names']:
        return
    linestyle = 'solid'
    color = config['method_colors'][method_name]
    label = config['method_names'][method_name]
    eps = subset['eps']
    if dp == 'none':
        eps = np.linspace(config['eps_range'][0], config['eps_range'][1])
        plt.hlines(y=subset[metric].mean(), xmin=config['eps_range'][0], xmax=config['eps_range'][1],
                   linestyle=linestyle, color=color, label=label)
        plt.fill_between(eps, subset[metric] - subset[f"{metric}_h"], subset[metric] + subset[f"{metric}_h"],
                         color=color,
                         alpha=0.2)
    else:
        mask = eps <= config['eps_range'][1]
        plt.scatter(eps[mask], subset[metric][mask], linestyle=linestyle, color=color, label=label)
        plt.plot(eps[mask], subset[metric][mask], linestyle=linestyle, color=color, label=label)

        plt.fill_between(eps[mask], subset[metric][mask] - subset[f"{metric}_h"][mask],
                         subset[metric][mask] + subset[f"{metric}_h"][mask], color=color,
                         alpha=0.2)


def finalize_plot(metric, folder, dataset=""):
    plt.xlabel('ε')
    plt.ylabel(metrics_dict[metric])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{dataset}.png"))
    plt.clf()


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 1:
        CONFIG['datasets_folders'] = [f"{sys.argv[1]}/accuracy"]
    process_datasets(CONFIG)
    create_legend_image(CONFIG)
