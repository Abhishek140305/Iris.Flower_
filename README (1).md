# Iris Flower Unsupervised Learning & Clustering Project

An end-to-end Python pipeline demonstrating **Unsupervised Machine Learning**, **Dimensionality Reduction**, and **Model Evaluation** on the classical Iris dataset.

---

## 📌 Project Overview

This project covers key unsupervised learning concepts:
1. **Feature Scaling**: Standardization using `StandardScaler`.
2. **Dimensionality Reduction**: 2D projection using Principal Component Analysis (`PCA`).
3. **Optimal Cluster Selection**: 
   - **Elbow Method** (Sum of Squared Errors / Inertia curve)
   - **Silhouette Analysis** across $k \in [2, 10]$
4. **Clustering**: Partitioning scaled 4D features using **K-Means** ($k=3$) and projecting cluster centroids into 2D PCA space.
5. **Model Evaluation**:
   - **Intrinsic Unsupervised Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
   - **Extrinsic Ground-Truth Metrics**: Adjusted Rand Index (ARI), Contingency Matrix.
6. **Model Serialization**: Saving and reloading fitted transformers and cluster models using both `joblib` and `pickle`.

---

## 🗂 Project Structure

```text
.
├── iris_unsupervised_workflow.py    # Main end-to-end executable script & modular library
├── models/                         # Serialized pipeline artifacts (generated on run)
│   ├── scaler.joblib / scaler.pkl
│   ├── pca.joblib / pca.pkl
│   └── kmeans.joblib / kmeans.pkl
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository / Create working directory
```bash
git clone <repository_url>
cd iris-unsupervised-clustering
```

### 2. Set up a virtual environment (recommended)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### `requirements.txt`
```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 🚀 Usage

### Run as a Standalone Script
To execute the complete end-to-end analysis, display diagnostic plots, print evaluation metrics, and export serialized model artifacts:

```bash
python iris_unsupervised_workflow.py
```

### Import as a Modular Library (e.g., in Jupyter Notebooks)
```python
import numpy as np
from iris_unsupervised_workflow import (
    load_and_preprocess,
    fit_pca,
    select_cluster_count,
    evaluate_clustering,
    save_models,
    verify_reloaded_predictions,
    FINAL_K
)
from sklearn.cluster import KMeans

# 1. Preprocess & Project
X, y, X_scaled, scaler = load_and_preprocess()
X_pca, pca = fit_pca(X_scaled)

# 2. Fit K-Means
kmeans = KMeans(n_clusters=FINAL_K, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)

# 3. Evaluate & Persist
metrics = evaluate_clustering(X_scaled, y, labels)
output_dir = save_models(scaler, pca, kmeans)
```

---

## 📊 Methodology & Workflow

### 1. Preprocessing & Scaling
Because features such as petal length and sepal width have varying variances and scales, `StandardScaler` is fitted to transform features into zero-mean, unit-variance distributions:
$$z = \frac{x - \mu}{\sigma}$$

### 2. Dimensionality Reduction (PCA)
- Reduces 4 continuous features (`sepal length`, `sepal width`, `petal length`, `petal width`) to 2 orthogonal principal components ($PC_1, PC_2$).
- Captures over 95% of cumulative explained variance for unambiguous 2D cluster visualization.

### 3. Optimal Cluster Selection
- **Elbow Method**: Detects the diminishing returns in inertia reduction past $k=3$.
- **Silhouette Coefficient**: Measures sample cohesion vs. separation across cluster boundaries:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

### 4. Clustering in Native 4D Space
K-Means is fitted directly on the 4-dimensional scaled feature space rather than PCA-compressed coordinates, preserving full geometric variance. Cluster centers are subsequently projected into PCA coordinates for 2D visualization:
```python
centroids_pca = pca.transform(kmeans.cluster_centers_)
```

---

## 📈 Evaluation Metrics Summary

| Metric | Target | Description | Typical Iris Value ($k=3$) |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | Higher $\to 1.0$ | Intra-cluster distance vs. nearest-cluster distance | $\approx 0.46$ |
| **Davies-Bouldin Index** | Lower $\to 0.0$ | Ratio of within-cluster scatter to separation | $\approx 0.83$ |
| **Calinski-Harabasz Index** | Higher | Between-cluster variance to within-cluster variance | $\approx 241.9$ |
| **Adjusted Rand Index (ARI)** | Higher $\to 1.0$ | Agreement between cluster labels and ground-truth species | $\approx 0.62$ |

*Note: ARI and the Contingency Matrix use ground-truth species strictly as post-hoc diagnostic validation—labels are never provided to the clustering algorithm.*

---

## 💾 Model Persistence & Verification

Fitted estimators (`StandardScaler`, `PCA`, and `KMeans`) are serialized to disk in both formats:
- **`joblib`**: Optimized for numerical NumPy arrays and large Scikit-Learn estimators.
- **`pickle`**: Standard Python object serialization.

Inference on new observations requires scaling before predicting cluster labels:
```python
import joblib

scaler = joblib.load("models/scaler.joblib")
kmeans = joblib.load("models/kmeans.joblib")

sample = [[5.1, 3.5, 1.4, 0.2]]  # Unscaled raw measurements
predicted_cluster = kmeans.predict(scaler.transform(sample))
print(f"Predicted Cluster: {predicted_cluster[0]}")
```

---

## 📄 License
Distributed under the MIT License.
