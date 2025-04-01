import matplotlib.pyplot as plt
import numpy as np
import scipy

plt.rcParams.update({'font.size': 16})


def set_seed(seed):
    np.random.seed(seed)


def mean_confidence_interval(vals, confidence=0.95):
    a = 1.0 * np.array(vals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def plot_clusters(centroids, values):
    plt.clf()
    distances = distance_matrix_squared(values, centroids)
    associations = np.argmin(distances, axis=1)
    colors = ['r', 'g', 'b', 'y', 'darkgray', 'cyan', 'pink', 'orange', 'purple', 'olive', 'gray', 'brown', 'teal',
              'yellowgreen', 'lightcoral', 'lightpink', 'peru', 'tomato', 'gold', 'magenta']
    k = centroids.shape[0]
    for cluster in range(k):
        vals = values[associations == cluster]
        plt.scatter(vals[:, 0], vals[:, 1], color=colors[cluster % len(colors)])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black')


def distance_matrix_squared(X, Y):
    return np.sum((X[:, np.newaxis] - Y) ** 2, axis=2)
