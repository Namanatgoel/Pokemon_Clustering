"""
Orchestration order:
    1. Load & preprocess data       (data_preprocessing)
    2. PCA dimensionality reduction (dimensionality_reduction)
    3. K-Means clustering           (models)
    4. Automated DBSCAN clustering  (models)
    5. Save visualisation           (visualization)

Usage:
    python main.py --data pokemon.csv
    python main.py --data /absolute/path/to/pokemon.csv --output outputs/clustering_visualization.png
"""

import argparse
import sys

from src.data_preprocessing import load_and_preprocess
from src.dimensionality_reduction import apply_pca
from src.models import run_kmeans, run_dbscan
from src.visualization import plot_and_save


PIPELINE_CONFIG: dict = {
    "default_csv": "pokemon.csv",
    "default_output": "outputs/clustering_visualization.png",
}


def parse_args():
    """Parse command-line arguments.
    Returns:
        Namespace with attributes:
            - data (str): Path to the pokemon CSV file.
            - output (str): Path for the output PNG.
    """
    parser = argparse.ArgumentParser(
        description="Pokémon K-Means & DBSCAN Clustering"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=PIPELINE_CONFIG["default_csv"],
        help="Path to pokemon.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=PIPELINE_CONFIG["default_output"],
        help="Output path for the cluster visualisation PNG (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    """
    Returns:
        None
    Raises:
        SystemExit: On unrecoverable errors (file not found, bad data shape).
    """
    args: argparse.Namespace = parse_args()

    print("[1/5] Loading and preprocessing data...")
    try:
        X_scaled, feature_names = load_and_preprocess(args.data)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[ERROR] Preprocessing failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"      Samples after preprocessing: {X_scaled.shape[0]}")
    print(f"      Features: {feature_names}")

    print("[2/5] Applying PCA (n_components=2)...")
    X_pca, var_ratio, pca_components = apply_pca(X_scaled)
    print(f"      PC1 explained variance: {var_ratio[0]:.3f}")
    print(f"      PC2 explained variance: {var_ratio[1]:.3f}")
    print(f"      Cumulative: {sum(var_ratio):.3f}")

    print("[3/5] Running K-Means (k=3, init=k-means++, random_state=42)...")
    km_labels, km_centers = run_kmeans(X_scaled)
    import numpy as np
    unique, counts = np.unique(km_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"      Cluster {cluster_id}: {count} Pokémon")

    print("[4/5] Running DBSCAN with automated ε derivation...")
    db_labels, optimal_eps = run_dbscan(X_pca)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = int(np.sum(db_labels == -1))
    print(f"      Derived ε: {optimal_eps:.4f}")
    print(f"      Clusters found (excl. noise): {n_clusters}")
    print(f"      Noise points: {n_noise}")

    print("[5/5] Generating and saving visualisation...")
    plot_and_save(
        X_pca=X_pca,
        kmeans_labels=km_labels,
        kmeans_centers=km_centers,
        dbscan_labels=db_labels,
        pca_components=pca_components,
        feature_names=feature_names,
        output_path=args.output,
    )
    print("\n Pipeline complete.")

if __name__ == "__main__":
    main()