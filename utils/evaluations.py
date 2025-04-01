"""
This module provides evaluation metrics for clustering algorithms.

It implements various metrics to assess clustering quality, including:
- Normalized Intra-cluster Variance (NICV)
- Between-Cluster Sum of Squares (BCSS)
- Empty cluster detection

The metrics help quantify both the compactness of clusters (how close points within
each cluster are to each other) and the separation between different clusters
(how well-distinguished the clusters are from one another).
"""

import numpy as np

from utils import distance_matrix_squared


def evaluate(centroids, values):
    """
    Evaluates the quality of a clustering solution using multiple metrics.
    
    This function computes several clustering evaluation metrics:
    1. Normalized Intra-cluster Variance (NICV): Measures the average variance within clusters
    2. Between-Cluster Sum of Squares (BCSS): Measures the separation between clusters
    3. Empty Clusters: Counts clusters with no assigned points
    
    Parameters
    ----------
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points to be clustered, shape (n_samples, n_features)
        
    Returns
    -------
    dict
        Dictionary containing the computed metrics:
        - 'Normalized Intra-cluster Variance (NICV)': float
        - 'Between-Cluster Sum of Squares (BCSS)': float
        - 'Empty Clusters': int
        
    Raises
    ------
    ValueError
        If no non-empty clusters are detected in the solution
    """
    distances = distance_matrix_squared(values, centroids)
    associations = get_cluster_associations(distances)
    non_empty_clusters = np.unique(associations).size

    if non_empty_clusters == 0:
        raise ValueError("No non-empty clusters detected.")
    empty_clusters = centroids.shape[0] - non_empty_clusters
    return {
        "Normalized Intra-cluster Variance (NICV)": evaluate_NICV(associations, centroids, values),
        "Between-Cluster Sum of Squares (BCSS)": evaluate_BCSS(associations, centroids, values),
        "Empty Clusters": empty_clusters
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
