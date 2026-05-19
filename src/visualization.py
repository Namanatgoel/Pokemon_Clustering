"""
Generates and saves a combined clustering visualisation in the 2-D PCA
space.  
The figure contains:
  - Left panel : K-Means cluster scatter with cluster centroids.
  - Right panel: DBSCAN cluster scatter with noise points highlighted.
  - Feature loading vectors (biplot arrows) overlaid on the K-Means panel.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ_CONFIG: dict = {
    "output_path": "outputs/clustering_visualization.png",
    "arrow_scale": 3.5,
    "figsize": (16, 7),
    "dpi": 150,
}

def plot_and_save(
    X_pca: np.ndarray,
    kmeans_labels: np.ndarray,
    kmeans_centers: np.ndarray,
    dbscan_labels: np.ndarray,
    pca_components: np.ndarray,
    feature_names: list[str],
    output_path: str = VIZ_CONFIG["output_path"],
):
    """
    Left panel shows K-Means clusters with centroid markers and feature
    loading arrows.  Right panel shows DBSCAN cluster assignments.
    
    Args:
        X_pca: 2-D array of shape (n_samples, 2) — PCA-transformed data.
        kmeans_labels: 1-D integer array of K-Means cluster assignments.
        kmeans_centers: 2-D array of shape (n_clusters, n_features) —
            K-Means centroids in the *original* scaled feature space.
            These are projected to 2-D PCA space internally.
        dbscan_labels: 1-D integer array of DBSCAN assignments; -1 = noise.
        pca_components: 2-D array of shape (2, n_features) — PCA loading
            vectors (rows = PC1, PC2).
        feature_names: Ordered list of feature names corresponding to
            columns of pca_components.
        output_path: Absolute or relative path at which the PNG will be
            saved.  Parent directory is created if it does not exist.

    Returns:
        None

    Raises:
        ValueError: If X_pca does not have exactly 2 columns.
        OSError: If the output directory cannot be created.
    """
    if X_pca.shape[1] != 2:
        raise ValueError(
            f"X_pca must have exactly 2 columns; received shape {X_pca.shape}."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    PC1: np.ndarray = pca_components[0]
    PC2: np.ndarray = pca_components[1]
    scale: float = VIZ_CONFIG["arrow_scale"]

    fig, axes = plt.subplots(1, 2, figsize=VIZ_CONFIG["figsize"])

    # Left panel: K-Means
    ax_km = axes[0]
    scatter_km = ax_km.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=kmeans_labels, cmap="viridis", s=50, alpha=0.6,
        label="Data points",
    )

    # Project centroids from scaled feature space into PCA 2-D space
    centers_pca: np.ndarray = kmeans_centers @ pca_components.T
    ax_km.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        color="red", marker="X", s=250, zorder=5, label="Centroids",
    )

    # Feature loading arrows
    for i, fname in enumerate(feature_names):
        ax_km.arrow(
            0, 0, PC1[i] * scale, PC2[i] * scale,
            color="crimson", alpha=0.75, head_width=0.12, length_includes_head=True,
        )
        ax_km.text(
            PC1[i] * scale * 1.12, PC2[i] * scale * 1.12,
            fname, color="crimson", fontsize=8,
        )

    ax_km.set_xlabel("PC1")
    ax_km.set_ylabel("PC2")
    ax_km.set_title("K-Means Clusters with Feature Loading Vectors")
    ax_km.grid(True, linestyle="--", alpha=0.4)
    ax_km.legend(fontsize=8)

    # Right panel: DBSCAN
    ax_db = axes[1]
    unique_labels: np.ndarray = np.unique(dbscan_labels)
    cmap_db = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))

    for label in unique_labels:
        mask: np.ndarray = dbscan_labels == label
        color = "black" if label == -1 else cmap_db(label)
        marker = "x" if label == -1 else "o"
        point_label = "Noise" if label == -1 else f"Cluster {label}"
        ax_db.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[color], s=50, alpha=0.6, marker=marker, label=point_label,
        )

    ax_db.set_xlabel("PC1")
    ax_db.set_ylabel("PC2")
    ax_db.set_title("DBSCAN Clusters (auto-ε)")
    ax_db.grid(True, linestyle="--", alpha=0.4)
    ax_db.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Plot saved → {os.path.abspath(output_path)}")