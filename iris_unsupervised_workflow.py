"""End-to-end unsupervised learning workflow for the Iris dataset.

The script is also notebook-friendly: import individual functions, or run it
directly to execute the complete analysis and save fitted model artifacts.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    contingency_matrix,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
FINAL_K = 3  # Kept explicit for the known three Iris species.


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series, np.ndarray, StandardScaler]:
    """Load Iris, retain labels for validation, and standardize numeric features."""
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()  # Not used to train clustering models.

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, y, X_scaled, scaler


def fit_pca(X_scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Fit a two-component PCA model and report explained variance."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    variance = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {variance}")
    print(f"Cumulative explained variance: {variance.sum():.4f}")
    return X_pca, pca


def plot_pca_projection(X_pca: np.ndarray) -> None:
    """Plot unlabeled observations in the first two PCA dimensions."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], s=65, color="steelblue")
    plt.title("Iris Dataset: 2D PCA Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.show()


def select_cluster_count(X_scaled: np.ndarray, k_values: range = range(1, 11)) -> tuple[int, dict[int, float]]:
    """Plot inertia and silhouette curves; return silhouette-selected k and scores.

    K-means is fitted on scaled original features, rather than the 2D plotting
    representation, so clustering retains all of the available information.
    """
    inertias: list[float] = []
    silhouette_scores: dict[int, float] = {}

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        if k >= 2:
            silhouette_scores[k] = silhouette_score(X_scaled, labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_values), inertias, marker="o")
    axes[0].set(title="Elbow Method", xlabel="Number of clusters (k)", ylabel="Inertia")
    axes[0].grid(alpha=0.3)

    axes[1].plot(list(silhouette_scores), list(silhouette_scores.values()), marker="o", color="darkorange")
    axes[1].set(title="Silhouette Scores", xlabel="Number of clusters (k)", ylabel="Silhouette score")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    selected_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Silhouette-selected k: {selected_k} (score={silhouette_scores[selected_k]:.4f})")
    return selected_k, silhouette_scores


def plot_clusters(X_pca: np.ndarray, cluster_labels: np.ndarray, kmeans: KMeans, pca: PCA) -> None:
    """Visualize cluster assignments and projected K-means centroids."""
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plot_data = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": cluster_labels.astype(str)})

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=plot_data, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=70)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X", s=230, label="Centroids")
    plt.title("K-Means Clusters in PCA Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()


def evaluate_clustering(X_scaled: np.ndarray, y: pd.Series, labels: np.ndarray) -> dict[str, Any]:
    """Print and return intrinsic metrics plus post-hoc label agreement metrics."""
    metrics: dict[str, Any] = {
        "silhouette_score": silhouette_score(X_scaled, labels),
        "davies_bouldin_index": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz_index": calinski_harabasz_score(X_scaled, labels),
        "adjusted_rand_index": adjusted_rand_score(y, labels),
        "contingency_matrix": contingency_matrix(y, labels),
    }
    print("\nFinal model evaluation")
    for name in ("silhouette_score", "davies_bouldin_index", "calinski_harabasz_index", "adjusted_rand_index"):
        print(f"{name}: {metrics[name]:.4f}")
    print("Contingency matrix (rows=true species, columns=clusters):")
    print(metrics["contingency_matrix"])
    return metrics


def save_models(scaler: StandardScaler, pca: PCA, kmeans: KMeans, output_dir: str | Path = "models") -> Path:
    """Persist every fitted transformer/model using both joblib and pickle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    models = {"scaler": scaler, "pca": pca, "kmeans": kmeans}
    for name, model in models.items():
        joblib.dump(model, output_path / f"{name}.joblib")
        with (output_path / f"{name}.pkl").open("wb") as file:
            pickle.dump(model, file)
    return output_path


def verify_reloaded_predictions(X_new: np.ndarray, artifact_dir: str | Path = "models") -> None:
    """Demonstrate that joblib and pickle artifacts give identical predictions."""
    artifact_path = Path(artifact_dir)
    # New samples must have the same four raw feature columns as the training data.
    scaler_joblib = joblib.load(artifact_path / "scaler.joblib")
    kmeans_joblib = joblib.load(artifact_path / "kmeans.joblib")
    with (artifact_path / "scaler.pkl").open("rb") as file:
        scaler_pickle = pickle.load(file)
    with (artifact_path / "kmeans.pkl").open("rb") as file:
        kmeans_pickle = pickle.load(file)

    predictions_joblib = kmeans_joblib.predict(scaler_joblib.transform(X_new))
    predictions_pickle = kmeans_pickle.predict(scaler_pickle.transform(X_new))
    assert np.array_equal(predictions_joblib, predictions_pickle), "Reloaded model predictions differ."
    print(f"Reload verification passed. Predictions: {predictions_joblib}")


def main() -> None:
    """Run the complete Iris clustering analysis."""
    sns.set_theme(style="whitegrid")
    X, y, X_scaled, scaler = load_and_preprocess()
    X_pca, pca = fit_pca(X_scaled)
    plot_pca_projection(X_pca)

    selected_k, _ = select_cluster_count(X_scaled)
    print(f"Using k={FINAL_K} for the final model (silhouette selection returned k={selected_k}).")
    kmeans = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    plot_clusters(X_pca, labels, kmeans, pca)
    evaluate_clustering(X_scaled, y, labels)
    artifact_dir = save_models(scaler, pca, kmeans)

    # Example unseen raw feature rows, ordered as in the Iris feature matrix.
    new_samples = np.array([[5.1, 3.5, 1.4, 0.2], [6.5, 3.0, 5.2, 2.0]])
    verify_reloaded_predictions(new_samples, artifact_dir)


if __name__ == "__main__":
    main()
