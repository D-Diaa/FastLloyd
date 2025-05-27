"""
This module provides evaluation metrics for clustering algorithms.

It implements various metrics to assess clustering quality, including:
- Normalized Intra-cluster Variance (NICV)
- Between-Cluster Sum of Squares (BCSS)
- Empty cluster detection
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Dunn Index
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def safe_metric(func, default, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Warning: {func.__name__} couldn't be calculated: {e}")
        return default


def evaluate(centroids, values, gt_centroids):
    """
    Evaluates the quality of a clustering solution using multiple metrics.

    This function computes several clustering evaluation metrics:
    1. Normalized Intra-cluster Variance (NICV): Measures the average variance within clusters
    2. Between-Cluster Sum of Squares (BCSS): Measures the separation between clusters
    3. Empty Clusters: Counts clusters with no assigned points
    4. Silhouette Score: Measures how well each object lies within its cluster
    5. Davies-Bouldin Index: Ratio of within-cluster and between-cluster distances
    6. Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion
    7. Dunn Index: Ratio of min between-cluster distance to max within-cluster distance
    8. Mean Squared Error: Average squared distance between assigned centroids and ground truth centroids

    Parameters
    ----------
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points to be clustered, shape (n_samples, n_features)
    gt_centroids : numpy.ndarray
        Array of ground truth centroids, shape (n_clusters, n_features)

    Returns
    -------
    dict
        Dictionary containing the computed metrics:
        - 'Normalized Intra-cluster Variance (NICV)': float
        - 'Between-Cluster Sum of Squares (BCSS)': float
        - 'Empty Clusters': int
        - 'Silhouette Score': float
        - 'Davies-Bouldin Index': float
        - 'Calinski-Harabasz Index': float
        - 'Dunn Index': float
        - 'Mean Squared Error': float

    Raises
    ------
    ValueError
        If no non-empty clusters are detected in the solution
    """
    distances = cdist(values, centroids)
    associations = get_cluster_associations(distances)
    non_empty_clusters = np.unique(associations).size

    if non_empty_clusters == 0:
        raise ValueError("No non-empty clusters detected.")

    if non_empty_clusters < 2:
        silhouette = -1
        davies_bouldin = np.inf
        calinski_harabasz = 0
        dunn = 0
    else:
        silhouette = safe_metric(silhouette_score, -1, values, associations)
        davies_bouldin = safe_metric(davies_bouldin_score, np.inf, values, associations)
        calinski_harabasz = safe_metric(calinski_harabasz_score, 0, values, associations)
        dunn = safe_metric(evaluate_dunn_index, 0, associations, values)

    empty_clusters = centroids.shape[0] - non_empty_clusters
    cost_matrix = cdist(centroids, gt_centroids)
    # Solve the assignment problem to minimize total squared error
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Extract the matched squared distances and average them
    matched_sq_dists = cost_matrix[row_ind, col_ind]
    mse = matched_sq_dists.mean()
    return {
        "Normalized Intra-cluster Variance (NICV)": evaluate_NICV(associations, centroids, values),
        "Between-Cluster Sum of Squares (BCSS)": evaluate_BCSS(associations, centroids, values),
        "Empty Clusters": empty_clusters,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": davies_bouldin,
        "Calinski-Harabasz Index": calinski_harabasz,
        "Dunn Index": dunn,
        "Mean Squared Error": mse
    }


def get_cluster_associations(distances):
    """
    Assigns each data point to its nearest cluster based on distance matrix.

    Parameters
    ----------
    distances : numpy.ndarray
        Square distance matrix between points and centroids,
        shape (n_samples, n_clusters)

    Returns
    -------
    numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
        Each element is the index of the closest centroid to that point
    """
    associations = np.argmin(distances, axis=1)
    return associations


def evaluate_NICV(associations, centroids, values):
    """
    Calculates the Normalized Intra-cluster Variance (NICV).

    NICV is the Within-Cluster Sum of Squares (WCSS) normalized by the number
    of data points. It represents the average variance of points within their
    clusters, with lower values indicating more compact clusters.

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)

    Returns
    -------
    float
        The NICV value (WCSS divided by number of samples)
    """
    return evaluate_WCSS(associations, centroids, values) / values.shape[0]


def evaluate_WCSS(associations, centroids, values):
    """
    Calculates the Within-Cluster Sum of Squares (WCSS).

    WCSS measures the compactness of clusters by summing the squared distances
    between each point and its assigned cluster centroid. Lower values indicate
    more compact clusters.

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)

    Returns
    -------
    float
        The WCSS value - sum of squared distances between points and their centroids
    """
    return sum([np.sum((values[associations == cluster] - centroids[cluster]) ** 2) for cluster in
                range(centroids.shape[0]) if np.sum(associations == cluster) > 0])


def evaluate_BCSS(associations, centroids, values):
    """
    Calculates the Between-Cluster Sum of Squares (BCSS).

    BCSS measures the separation between clusters by summing the weighted squared
    distances between each cluster centroid and the overall data centroid. Higher
    values indicate better-separated clusters.

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)

    Returns
    -------
    float
        The BCSS value - weighted sum of squared distances between centroids
        and the overall centroid
    """
    overall_centroid = np.mean(values, axis=0)
    return sum(
        [(np.linalg.norm(centroids[cluster] - overall_centroid) ** 2) * np.sum(associations == cluster) for cluster in
         range(centroids.shape[0])])


def evaluate_dunn_index(associations, values):
    """
    Calculates the Dunn Index for a clustering.

    The Dunn index is the ratio of the minimum inter-cluster distance to the
    maximum intra-cluster distance. Higher values indicate better clustering.

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)

    Returns
    -------
    float
        The Dunn index
    """
    unique_clusters = np.unique(associations)
    n_clusters = unique_clusters.size

    # Check if we have more than one cluster
    if n_clusters < 2:
        return np.nan

    # Calculate minimum inter-cluster distance
    min_inter_dist = float('inf')
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = values[associations == unique_clusters[i]]
            cluster_j = values[associations == unique_clusters[j]]

            # Calculate minimum distance between points in cluster i and j
            if len(cluster_i) > 0 and len(cluster_j) > 0:
                inter_dist = np.min(cdist(cluster_i, cluster_j))
                min_inter_dist = min(min_inter_dist, inter_dist)

    # Calculate maximum intra-cluster distance
    max_intra_dist = 0
    for i in range(n_clusters):
        cluster_i = values[associations == unique_clusters[i]]

        # Skip empty clusters
        if len(cluster_i) <= 1:
            continue

        # Calculate maximum distance between points in the same cluster
        intra_dist = np.max(cdist(cluster_i, cluster_i))
        max_intra_dist = max(max_intra_dist, intra_dist)

    # Handle edge cases
    if max_intra_dist == 0 or min_inter_dist == float('inf'):
        return np.nan

    return min_inter_dist / max_intra_dist
