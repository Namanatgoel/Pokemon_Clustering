"""
Handles loading, feature extraction, imputation, and scaling of the
Pokémon dataset for downstream clustering and dimensionality reduction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: list[str] = [
    "hp", "attack", "defense",
    "sp_attack", "sp_defense", "speed",
    "height_m", "weight_kg",
]


def load_and_preprocess(csv_path: str):
    """Load the Pokémon CSV, extract numerical features, impute missing values,
    and scale with StandardScaler.

    Args:
        csv_path: Absolute or relative path to the pokemon.csv file.

    Returns:
        A tuple of:
            - X_scaled: 2-D float64 array of shape (n_samples, 8) after
              StandardScaler normalisation.
            - feature_names: Ordered list of feature column names matching
              the columns of X_scaled.

    Raises:
        FileNotFoundError: If csv_path does not point to a readable file.
        KeyError: If any column listed in FEATURE_COLUMNS is absent from the
            loaded DataFrame.
    """
    df: pd.DataFrame = pd.read_csv(csv_path)

    missing: list[str] = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following required columns are absent from the dataset: {missing}"
        )

    X: pd.DataFrame = df[FEATURE_COLUMNS].copy()

    for col in ["height_m", "weight_kg"]:
        median_val: float = X[col].median()
        X[col] = X[col].fillna(median_val)

    X = X.dropna()

    scaler: StandardScaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X.values)

    return X_scaled, FEATURE_COLUMNS