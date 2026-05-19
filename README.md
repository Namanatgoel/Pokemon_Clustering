# Pokémon Clustering Pipeline

## 1. Project Title & Context

This repository implements an unsupervised clustering pipeline for the Pokémon
dataset.  Two algorithms are applied — K-Means and DBSCAN — in a 2-D space
derived from Principal Component Analysis.  The pipeline corrects two
deficiencies present in the original experimental notebook:

- **Duplicate feature column**: `speed` appeared twice in the feature list,
  artificially inflating its contribution to both principal components.
- **Hard-coded DBSCAN ε**: `eps=0.8` collapsed all points into a single
  cluster.  The production pipeline derives ε programmatically from the
  maximum-curvature (elbow) point of the sorted k-NN distance graph.

---

## 2. System Architecture

```
Pokemon_Clustering/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Feature extraction, imputation, scaling
│   ├── dimensionality_reduction.py # PCA (n_components=2)
│   ├── models.py                  # K-Means + automated DBSCAN
│   └── visualization.py           # Biplot scatter, saves PNG
├── main.py                        # Pipeline orchestrator
├── outputs/
│   └── clustering_visualization.png  # Generated at runtime
└── README.md
```

---

## 3. Engineering Optimisations

### 3.1 Automated ε Identification (DBSCAN)

The original notebook used a fixed `eps=0.8`, which produced only one cluster
and 20 noise points — an uninformative result.

The production pipeline uses the following algorithm implemented in
`src/models._compute_optimal_epsilon`:

1. Fit `sklearn.neighbors.NearestNeighbors(n_neighbors=4)` on the 2-D PCA
   array.
2. Extract the distance to the 4th nearest neighbour for every point.
3. Sort these distances in ascending order to form the k-NN distance curve.
4. For each point on the curve, compute its perpendicular distance to the
   straight line joining the first and last points of the curve.
5. The index of the maximum perpendicular distance identifies the elbow.
6. The distance value at that index is used as ε.

This yields a data-driven ε rather than an arbitrary constant, ensuring
DBSCAN separates meaningful density regions.

### 3.2 Median Imputation

`height_m` and `weight_kg` contain missing values in the source CSV.
Median imputation is applied column-wise before `StandardScaler` is fitted,
preventing information loss from row-wise `dropna`.

---

## 4. Execution Instructions

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run

```bash
# Assumes pokemon.csv is in the working directory
python main.py

# Explicit paths
python main.py --data /path/to/pokemon.csv --output outputs/clustering_visualization.png
```

---
