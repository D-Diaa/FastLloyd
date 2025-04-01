"""
Utilities for loading, processing, and normalizing data for distributed computing applications.

This module provides functions for loading data from text files, splitting data among clients,
and normalizing values, with optional fixed-point conversion support.
"""

import numpy as np

from data_io import to_fixed


def load_txt(path: str):
    """
    Load numerical data from a text file, skipping lines containing 'x'.

    Args:
        path (str): Path to the text file containing numerical data

    Returns:
        np.ndarray: Array containing the loaded numerical values

    Example:
        >>> values = load_txt("data.txt")
        >>> print(values.shape)
        (100, 5)  # If file contains 100 rows of 5 numbers each
    """
    values_list = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "x" in line:
                continue
            values_list.append([float(x) for x in line.split()])
    values_arr = np.array(values_list)
    return values_arr


def shuffle_and_split(values, clients, proportions=None):
    """
    Randomly shuffle data and split it among multiple clients.

    Args:
        values (np.ndarray): Input data array to be split
        clients (int): Number of clients to split the data among
        proportions (list of float, optional): Relative proportions for splitting data.
            If None, data is split equally. Defaults to None.

    Returns:
        list of np.ndarray: List of data arrays, one for each client

    Example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> splits = shuffle_and_split(data, clients=2)
        >>> print([split.shape for split in splits])
        [(2, 2), (2, 2)]  # Each client gets 2 samples
    """
    if proportions is None:
        size = len(values) // clients
        sizes = [size for _ in range(clients - 1)]
    else:
        prop_sum = sum(proportions)
        total = values.shape[0]
        sizes = [int(proportions[i] / prop_sum * total) for i in range(clients - 1)]
    np.random.shuffle(values)
    st = 0
    value_lists = []
    for client in range(clients - 1):
        size = sizes[client]
        value_lists.append(values[st: st + size, :])
        st += size
    value_lists.append(values[st:, :])
    return value_lists


def normalize(values, fixed=False):
    """
    Normalize data to [-1, 1] range, with optional fixed-point conversion.

    This function applies min-max normalization to map values to [-1, 1].
    For columns with all equal values, the normalized value is set to 0.

    Args:
        values (np.ndarray): Input array to normalize
        fixed (bool, optional): Whether to convert to fixed-point representation.
            Defaults to False.

    Returns:
        np.ndarray: Normalized values, optionally in fixed-point representation

    Example:
        >>> data = np.array([[1, 2], [3, 4]])
        >>> normalized = normalize(data)
        >>> print(normalized)
        [[-1. -1.]
         [ 1.  1.]]
    """
    mx = values.max(axis=0)
    mn = values.min(axis=0)
    normalized = np.zeros_like(values)
    empty = mx - mn == 0
    normalized[:, empty] = 0.5
    normalized[:, ~empty] = (values[:, ~empty] - mn[~empty]) / (mx[~empty] - mn[~empty])
    normalized = normalized * 2 - 1
    if fixed:
        return to_fixed(normalized)
    else:
        return normalized
