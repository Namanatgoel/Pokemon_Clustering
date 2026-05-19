"""
Provides K-Means and automated DBSCAN clustering models for the Pokémon
dataset.  The DBSCAN epsilon is derived programmatically from the elbow
of the k-nearest-neighbour distance graph.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

KMEANS_CONFIG: dict = {
    "n_clusters": 3,
    "init": "k-means++",
    "random_state": 42,
}

DBSCAN_CONFIG: dict = {
    "n_neighbors": 4,   # used by NearestNeighbors for epsilon search
    "min_samples": 5,
}


def run_kmeans(X: np.ndarray):
    """Fit K-Means with k=3 on the provided feature matrix.
    Args:
        X: 2-D float64 array of shape (n_samples, n_features).

    Returns:
        A tuple of:
            - labels: 1-D integer array of shape (n_samples,) containing
              cluster assignments in the range [0, 2].
            - cluster_centers: 2-D array of shape (3, n_features) holding
              the centroid coordinates.
    """
    km: KMeans = KMeans(
        n_clusters=KMEANS_CONFIG["n_clusters"],
        init=KMEANS_CONFIG["init"],
        random_state=KMEANS_CONFIG["random_state"],
        n_init="auto",
    )
    km.fit(X)
    return km.labels_, km.cluster_centers_

def _compute_optimal_epsilon(X: np.ndarray, n_neighbors: int):
    """Compute the optimal DBSCAN epsilon via the elbow of the sorted
    k-nearest-neighbour distance curve.

    The algorithm:
    1. Fit NearestNeighbors(n_neighbors) on X.
    2. Retrieve the distance to the n_neighbors-th neighbour for every point.
    3. Sort distances in ascending order.
    4. Identify the index of maximum curvature (elbow) using the perpendicular
       distance from each point on the curve to the line joining the first and
       last point.
    5. Return the distance value at the elbow index as epsilon.

    Args:
        X: 2-D float64 array of shape (n_samples, n_features).
        n_neighbors: Number of neighbours to use; matches DBSCAN_CONFIG.

    Returns:
        Optimal epsilon value (float) derived from the elbow point.
    """
    nbrs: NearestNeighbors = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Distance to the furthest of the n_neighbors neighbours
    kth_distances: np.ndarray = np.sort(distances[:, -1])

    # Perpendicular-distance method to locate elbow
    n_pts: int = len(kth_distances)
    x_coords: np.ndarray = np.arange(n_pts, dtype=float)
    y_coords: np.ndarray = kth_distances

    # Line from first to last point
    p1: np.ndarray = np.array([x_coords[0], y_coords[0]])
    p2: np.ndarray = np.array([x_coords[-1], y_coords[-1]])
    line_vec: np.ndarray = p2 - p1
    line_len: float = float(np.linalg.norm(line_vec))

    perp_distances: np.ndarray = np.empty(n_pts)
    for i in range(n_pts):
        point_vec = np.array([x_coords[i], y_coords[i]]) - p1
        proj_len = np.dot(point_vec, line_vec) / line_len
        proj = p1 + proj_len * (line_vec / line_len)
        perp_distances[i] = float(np.linalg.norm(np.array([x_coords[i], y_coords[i]]) - proj))

    elbow_idx: int = int(np.argmax(perp_distances))
    optimal_eps: float = float(kth_distances[elbow_idx])
    return optimal_eps


def run_dbscan(X: np.ndarray):
    """Fit DBSCAN with an automatically derived epsilon.

    Epsilon is computed at the maximum-curvature (elbow) point of the
    sorted 4-nearest-neighbour distance graph, then passed together with
    min_samples=5 to sklearn DBSCAN.

    Args:
        X: 2-D float64 array of shape (n_samples, n_features).

    Returns:
        A tuple of:
            - labels: 1-D integer array of shape (n_samples,) where -1
              denotes noise and non-negative integers denote cluster ids.
            - epsilon: The derived epsilon value used by DBSCAN.
    """
    eps: float = _compute_optimal_epsilon(X, DBSCAN_CONFIG["n_neighbors"])
    db: DBSCAN = DBSCAN(eps=eps, min_samples=DBSCAN_CONFIG["min_samples"])
    labels: np.ndarray = db.fit_predict(X)
    return labels, eps