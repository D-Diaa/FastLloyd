# Federated Clustering Utilities

A comprehensive package of utility functions and evaluation metrics supporting privacy-preserving federated clustering operations. This package provides essential tools for measuring clustering quality, visualizing results, implementing federated protocols, and performing common mathematical operations needed throughout the clustering process.

## Core Components

### Federation Protocols

The package implements two distinct protocols for federated clustering, each serving different use cases and deployment scenarios.

#### MPI Protocol (`mpi_proto`)

The Message Passing Interface (MPI) protocol enables distributed computation across multiple processes, making it suitable for real-world federated learning deployments. This protocol implements a complete federated learning workflow with the following characteristics:

```python
from utils.protocols import mpi_proto
from configs import Params

# Configure the clustering parameters
params = Params(
    num_clients=4,
    dp="gaussiananalytic",
    eps=0.1
)

# Run the MPI-based protocol
centroids, unassigned = mpi_proto(
    value_lists=client_data,  # List of data for each client
    params=params,
    method="masked"  # Use privacy-preserving computation
)
```

Key features of the MPI protocol include:
- True distributed computation across multiple processes
- Efficient communication through MPI primitives
- Support for both masked (privacy-preserving) and unmasked computation
- Built-in communication delay simulation for realistic testing
- Progress tracking and communication statistics
- Synchronized updates across all clients

The protocol follows this workflow in each iteration:
1. Clients compute local statistics (totals and counts)
2. Statistics are gathered at the server through MPI communication
3. Server processes the aggregated statistics and may add differential privacy noise
4. Results are broadcast back to all clients via MPI
5. Clients update their local centroids based on the aggregated results

#### Local Protocol (`local_proto`)

The local protocol simulates federated clustering within a single process, making it ideal for development, testing, and experimental validation:

```python
from utils.protocols import local_proto
from configs import Params

# Configure the clustering parameters
params = Params(
    num_clients=4,
    dp="gaussiananalytic",
    eps=0.1
)

# Run the local simulation protocol
centroids, unassigned = local_proto(
    value_lists=client_data,  # List of data for each client
    params=params,
    method="masked"  # Use privacy-preserving computation
)
```

The local protocol offers several advantages for development:
- Simulates complete federated learning workflow in memory
- Enables easy debugging and development iteration
- Tracks convergence through centroid movement
- Monitors unassigned points across iterations
- Maintains a history of centroid positions
- Perfect for testing privacy mechanisms and constraint methods

The simulation follows these steps in each iteration:
1. Each simulated client computes local statistics
2. The server aggregates these statistics in memory
3. Clients update their centroids using the aggregated statistics
4. Progress is tracked through centroid movement metrics
5. Unassigned points (too far from any centroid) are monitored

### Distance Calculations

The package implements efficient distance calculations that serve as the foundation for clustering operations. The primary function `distance_matrix_squared` computes pairwise squared Euclidean distances between two sets of points, optimized for the repeated calculations needed in k-means clustering:

```python
from utils import distance_matrix_squared

# Calculate distances between points and centroids
distances = distance_matrix_squared(data_points, centroids)
```

### Evaluation Metrics

The package provides several metrics to assess clustering quality through the `evaluate` function. These metrics help understand both the compactness of clusters and the separation between them:

```python
from utils import evaluate

metrics = evaluate(centroids, data_points)
# Returns a dictionary with:
# - Normalized Intra-cluster Variance (NICV)
# - Between-Cluster Sum of Squares (BCSS)
# - Number of Empty Clusters
```

Each metric provides different insights into clustering quality:

1. Normalized Intra-cluster Variance (NICV) measures how compact the clusters are, with lower values indicating tighter, more coherent clusters. It represents the average squared distance between points and their cluster centroids, normalized by the number of data points for fair comparison across datasets.

2. Between-Cluster Sum of Squares (BCSS) quantifies how well-separated the clusters are, with higher values indicating better cluster separation. It uses weighted distances between cluster centroids and the overall data centroid, where weights are based on cluster sizes to account for varying cluster populations.

3. Empty Clusters tracks clusters with no assigned points, helping identify potential issues with cluster initialization or privacy constraints.

### Visualization Tools

The package includes visualization capabilities through the `plot_clusters` function, which creates informative 2D plots of clustering results:

```python
from utils import plot_clusters

# Create a scatter plot of clusters with centroids
plot_clusters(centroids, data_points)
# - Points are colored by cluster assignment
# - Centroids are marked in black
# - Supports up to 20 distinct cluster colors
```

### Statistical Analysis

For analyzing experimental results across multiple runs, the package provides statistical tools:

```python
from utils import mean_confidence_interval

# Calculate mean and confidence interval for a set of measurements
mean, confidence_interval = mean_confidence_interval(
    values,
    confidence=0.95  # 95% confidence interval by default
)
```

This function enables:
- Assessment of clustering stability
- Comparison of different privacy mechanisms
- Evaluation of parameter impacts
- Reporting of results with statistical significance

### Reproducibility

The package ensures reproducible results through the `set_seed` function:

```python
from utils import set_seed

# Set random seed for reproducibility
set_seed(1337)  # Use the same seed for consistent results
```

## Understanding the Metrics

### NICV (Normalized Intra-cluster Variance)

NICV helps us understand how well points fit within their assigned clusters:

When NICV is low, it indicates that points are very close to their cluster centroids, forming compact and well-defined clusters. This suggests strong patterns in the data. Conversely, a high NICV might indicate that points are spread far from their centroids, suggesting either too few clusters or noisy data.

### BCSS (Between-Cluster Sum of Squares)

BCSS helps us evaluate how distinct our clusters are from each other:

A high BCSS indicates well-separated clusters with clear boundaries between different groups, suggesting a strong cluster structure in the data. A low BCSS might indicate significant cluster overlap, which could suggest too many clusters or a lack of clear structure in the data.

## Integration with Privacy-Preserving Features

These utilities are designed to work seamlessly with privacy-preserving mechanisms:

The distance calculations handle both raw and noise-added data effectively. The evaluation metrics remain meaningful even with differential privacy noise present. The visualization tools work equally well with both original and protected data. The federation protocols support both standard and privacy-preserving computation through their masked and unmasked modes.

## Complete Example Workflow

Here's a comprehensive example of using the utilities for federated clustering:

```python
from utils import set_seed, evaluate, plot_clusters, mean_confidence_interval
from utils.protocols import local_proto
from configs import Params

# Ensure reproducibility
set_seed(1337)

# Configure clustering parameters
params = Params(
    num_clients=4,
    dp="gaussiananalytic",
    eps=0.1,
    k=5  # number of clusters
)

# Run federated clustering simulation
centroids, unassigned = local_proto(
    value_lists=client_data,
    params=params,
    method="masked"
)

# Evaluate clustering quality
metrics = evaluate(centroids, client_data[0])

# Visualize the results
plot_clusters(centroids, client_data[0])

# Analyze stability across multiple runs
nicv_values = []
for _ in range(10):
    set_seed(1337 + _)
    trial_centroids, _ = local_proto(client_data, params, "masked")
    trial_metrics = evaluate(trial_centroids, client_data[0])
    nicv_values.append(trial_metrics["Normalized Intra-cluster Variance (NICV)"])

mean_nicv, ci = mean_confidence_interval(nicv_values)
print(f"Average NICV: {mean_nicv:.4f} ± {ci:.4f}")
```