"""
Applies sklearn PCA to reduce the scaled Pokémon feature matrix to 2
principal components for visualisation and clustering.
"""

import numpy as np
from sklearn.decomposition import PCA


CONFIG: dict = {
    "n_components": 2,
}


def apply_pca(
    X_scaled: np.ndarray,
):
    """Fit and apply PCA to reduce X_scaled to 2 principal components.

    Args:
        X_scaled: 2-D float64 array of shape (n_samples, n_features) produced
            by the preprocessing step.

    Returns:
        A tuple of:
            - X_pca: Transformed array of shape (n_samples, 2) in the 2-D PCA
              space.
            - explained_variance_ratio: 1-D array of length 2 containing the
              fraction of total variance explained by PC1 and PC2.
            - components: 2-D array of shape (2, n_features) whose rows are
              the principal-component loading vectors (eigenvectors).

    Raises:
        ValueError: If X_scaled has fewer than 2 columns.
    """
    if X_scaled.shape[1] < CONFIG["n_components"]:
        raise ValueError(
            f"X_scaled must have at least {CONFIG['n_components']} columns; "
            f"received shape {X_scaled.shape}."
        )

    pca: PCA = PCA(n_components=CONFIG["n_components"])
    X_pca: np.ndarray = pca.fit_transform(X_scaled)

    return X_pca, pca.explained_variance_ratio_, pca.components_