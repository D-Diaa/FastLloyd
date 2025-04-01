import numpy as np
from diffprivlib.mechanisms import GaussianAnalytic


class Params:
    """Configuration parameters for privacy-preserving federated clustering.
    
    This class manages parameters for both the clustering algorithm and its privacy
    mechanisms. It handles initialization of centroids, calculation of privacy budgets,
    and dynamic updates of clustering constraints.
    
    Key Features:
    - Supports both Laplace and Gaussian Analytic differential privacy mechanisms
    - Provides methods for privacy budget allocation and sensitivity calculation
    - Implements methods for constraining maximum distance for centroid updates
    - Handles initialization of cluster centroids with proximity constraints
    
    Attributes:
        seed (int): Random seed for reproducibility (default: 1337)
        data_size (int): Total number of data points (default: 1000)
        dim (int): Dimensionality of the data (default: 2)
        k (int): Number of clusters (default: 15)
        iters (int): Number of clustering iterations (default: 6)
        alpha (float): Parameter for controlling update constraints (default: 2.0)
        max_dist (float): Maximum allowed distance for updates (default: 1e9)
        num_clients (int): Number of clients in federated setup (default: 2)
        delay (int): Communication delay simulation (default: 0)
        rho (float): Privacy parameter for noise calibration (default: 0.225)
        eps (float): Privacy budget epsilon (default: 0)
        dp (str): Differential privacy mechanism ["none"|"laplace"|"gaussiananalytic"] (default: "none")
        method (str): Constraint method for updates (default: "none")
        post (str): Post-processing method ["none"|"fold"|"truncate"] (default: "none")
        fixed (bool): Whether to use fixed-point arithmetic (default: True)
        
    Constants:
        DEFAULT_BOUNDS (tuple): Default data bounds (-1, 1)
        MAX_RETRIES (int): Maximum retries for centroid initialization (100)
        MAX_ITERATIONS (int): Maximum allowed iterations (7)
        MIN_ITERATIONS (int): Minimum required iterations (2)
    """

    seed: int = 1337
    data_size: int = 1000
    dim: int = 2
    k: int = 15
    iters: int = 6
    alpha: float = 2.0
    max_dist: float = 1e9
    num_clients: int = 2
    delay: int = 0
    rho: float = 0.225
    eps: float = 0
    dp: str = "none"
    method: str = "none"
    post: str = "none"
    fixed: bool = True

    DEFAULT_BOUNDS = (-1, 1)
    MAX_RETRIES = 100
    MAX_ITERATIONS = 7
    MIN_ITERATIONS = 2

    def __init__(self, **kwargs):
        """Initialize parameters with optional overrides.
        
        Args:
            **kwargs: Keyword arguments to override default parameter values
            
        Raises:
            AttributeError: If an invalid parameter name is provided
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid parameter of {self.__class__.__name__}")
        self.attributes = [
            attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")
        ]

    def init_centroids(self):
        """Initialize centroids with cluster proximity processing.
        
        Implements a sophisticated centroid initialization strategy that ensures
        minimum distance between centroids while respecting domain bounds.
        The algorithm uses an adaptive proximity threshold that is halved
        if the current threshold is too restrictive.
        
        Originally from: https://github.com/IBM/differential-privacy-library/
        
        Returns:
            np.ndarray: Array of initialized centroids, or None if initialization fails
            
        Note:
            - Centroids are initialized within DEFAULT_BOUNDS
            - Maintains minimum distance between centroids using cluster_proximity
            - Uses binary search-like approach to find valid initialization
        """
        dims = self.dim
        k = self.k
        generator = np.random.RandomState(self.seed)
        bounds_processed = np.zeros(shape=(dims, 2))
        for dim in range(dims):
            bounds_processed[dim, :] = [self.DEFAULT_BOUNDS[1] - self.DEFAULT_BOUNDS[0], self.DEFAULT_BOUNDS[0]]

        cluster_proximity = np.min(bounds_processed[:, 0]) / 2.0
        while cluster_proximity > 0:
            centers = np.zeros(shape=(k, dims))
            cluster, retry = 0, 0
            while retry < self.MAX_RETRIES:
                if cluster >= k:
                    break
                temp_center = (generator.random(dims) * (bounds_processed[:, 0] - 2 * cluster_proximity) +
                               bounds_processed[:, 1] + cluster_proximity)
                if cluster == 0:
                    centers[0, :] = temp_center
                    cluster += 1
                    continue
                min_distance = ((centers[:cluster, :] - temp_center) ** 2).sum(axis=1).min()
                if np.sqrt(min_distance) >= 2 * cluster_proximity:
                    centers[cluster, :] = temp_center
                    cluster += 1
                    retry = 0
                else:
                    retry += 1
            if cluster >= k:
                return centers
            cluster_proximity /= 2.0
        return None

    def calculate_iters(self):
        """Calculate the optimal number of iterations based on privacy budget.
        
        Determines the number of iterations that can be performed while maintaining
        differential privacy guarantees. The calculation differs based on the
        privacy mechanism (Laplace or Gaussian) and the constraint method used.
        
        For Laplace mechanism: Based on the k-means convergence analysis from
        the IBM differential privacy library [SU16].
        
        For Gaussian mechanism: Considers different constraint methods and their
        impact on the privacy-utility trade-off.
        
        Note:
            Always ensures iterations are between MIN_ITERATIONS and MAX_ITERATIONS
        
        Raises:
            ValueError: If an invalid differential privacy mechanism or method is specified
        """
        if self.dp == "none":
            self.iters = self.MAX_ITERATIONS
        else:
            d = self.dim
            k = self.k
            n = self.data_size
            eps = self.eps
            delta = 1 / (n * np.log(n))
            rho = self.rho
            if self.dp == "laplace":
                """From: https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/models/k_means.py"""
                epsilon_m = np.sqrt(500 * (k ** 3) / (n ** 2) * (d + np.cbrt(4 * d * (rho ** 2))) ** 3)
                iters = eps / epsilon_m
            elif "gaussian" in self.dp:
                scale = GaussianAnalytic(epsilon=self.eps, delta=delta, sensitivity=1)._scale
                if self.method == "none":
                    iters = 0.004 * n ** 2 / (k ** 3 * d * (2 * rho + np.sqrt(d)) ** 2 * scale ** 2)
                else:
                    eta_d = 2 * np.sqrt(d) * self.get_frac()
                    iters = 0.004 * 4 * n ** 2 / (k ** 3 * eta_d ** 2 * (1 + np.sqrt(4 * d)) ** 2 * scale ** 2)
            else:
                raise ValueError(f"Invalid differential privacy mechanism: {self.dp}")
            self.iters = int(max(min(iters, self.MAX_ITERATIONS), self.MIN_ITERATIONS))

    def get_domain_info(self):
        """Calculate domain information for privacy mechanism calibration.
        
        Computes the domain diagonal (maximum possible distance between points)
        and the original sensitivity based on the privacy mechanism in use.
        
        Returns:
            dict: Dictionary containing:
                - domain_diagonal: Maximum possible distance between points
                - original_sensitivity: Base sensitivity for privacy mechanism
        
        Note:
            Assumes data bounds are [-1, 1] in all dimensions
        """
        # assuming bounds are [-1, 1]
        domain_diagonal = 2 * np.sqrt(self.dim)
        # sum sensitivity is at most 1, so original_sensitivity is 1
        if self.dp == "laplace":
            original_sensitivity = self.dim
        elif "gaussian" in self.dp:
            if self.method == "none":
                original_sensitivity = np.sqrt(self.dim)
            else:
                original_sensitivity = domain_diagonal
        else:
            original_sensitivity = domain_diagonal

        return {
            "domain_diagonal": domain_diagonal,
            "original_sensitivity": original_sensitivity,
        }

    def split_epsilon(self):
        """Split privacy budget between sum and count queries.
        
        Allocates the privacy budget (epsilon) between two types of queries:
        1. Sum queries for computing cluster totals/updates
        2. Count queries for tracking cluster sizes
        
        The split ratio depends on the privacy mechanism and constraint method.
        
        Returns:
            tuple: A tuple containing:
                - float: Epsilon for sum queries
                - float: Epsilon for count queries
        """
        eps_sum = self.dim
        eps_c = np.cbrt(4 * self.dim * self.rho ** 2)
        normaliser = self.eps / self.iters / (eps_sum + eps_c)
        return eps_sum * normaliser, eps_c * normaliser

    def update_maxdist(self, _iter):
        """Update the maximum allowed distance for centroid updates.
        
        Adjusts the maximum allowed distance for centroids based on:
        - The current iteration
        - The chosen constraint method
        - Domain characteristics
        - Privacy parameters
        
        This dynamic adjustment helps balance convergence and privacy requirements, by bounding the sensitivity.
        
        Args:
            _iter (int): Current iteration number
            
        Raises:
            ValueError: If an invalid constraint method is specified
        """
        # Initialize static parameters
        domain = self.get_domain_info()
        domain_diagonal = domain["domain_diagonal"]
        original_sensitivity = domain["original_sensitivity"]
        method = self.method
        if method == "none":
            self.max_dist = domain_diagonal
        else:
            frac = self.get_frac()
            if "stay" in method:
                self.max_dist = domain_diagonal * frac
            elif "diagonal_then" in method:
                if _iter == 0:
                    self.max_dist = domain_diagonal / 2
                else:
                    self.max_dist = domain_diagonal * frac
        if self.max_dist >= original_sensitivity:
            self.max_dist = domain_diagonal

    def get_frac(self):
        return self.alpha / (2 * self.k ** (1 / self.dim))

    def __getitem__(self, index):
        """Enable indexing access to parameters.
        
        Args:
            index: Index of the parameter to access
            
        Returns:
            Value of the parameter at the given index
        """
        return getattr(self, self.attributes[index])
