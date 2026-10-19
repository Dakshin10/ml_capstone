# 23CSE301 — Machine Learning Capstone

**End-to-End Machine Learning Pipeline: Regression (Review 1) & Clustering (Review 2)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Review%202%20Clustering%20Ready-brightgreen.svg)]()

[Overview](#1-project-overview) &nbsp;|&nbsp;
[Regression Track (R1)](#2-regression-track-review-1) &nbsp;|&nbsp;
[Clustering Track (R2)](#3-clustering-track-review-2) &nbsp;|&nbsp;
[Clustering Results](#4-clustering-benchmarks-and-personas) &nbsp;|&nbsp;
[Repository Structure](#5-repository-structure) &nbsp;|&nbsp;
[Quick Start](#6-installation-and-quick-start)

---

</div>


## 1. Project Overview

This sub-section of the **23CSE301 Machine Learning Capstone (Review 1)** implements five non-linear regression algorithms on the UCI Metro Interstate Traffic Volume dataset, predicting hourly westbound traffic volume on the I-94 Minnesota corridor.

This work is one part of a larger team capstone project. The shared pipeline contract (preprocessing steps, split strategy, leakage guard) is designed to be consistent with the rest of the team's regression and classification notebooks so that results can be compared and merged.

**Scope of this sub-section:**

| Property | Value |
| :--- | :--- |
| Task | Supervised Regression |
| Dataset | UCI Metro Interstate Traffic Volume |
| Target Variable | `traffic_volume` — Hourly vehicle count (vehicles/hr) |
| Algorithms | Decision Tree, Random Forest, Gradient Boosting, SVR, KNN |
| Evaluation | R2 Score, RMSE, MAE, 5-Fold Cross-Validation (top 2 models) |
| Notebook | `notebooks/regression.ipynb` |

---

## 2. Dataset

**Source:** [UCI ML Repository — Dataset #492: Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

**File:** `data/Metro_Interstate_Traffic_Volume.csv`

**Description:** Hourly westbound traffic volume records on I-94 at ATR station 301 near Minneapolis-St. Paul, MN, collected from 2012 to 2018. Weather and holiday conditions are included as contextual features.

### Raw Columns

| Column | Type | Description |
| :--- | :--- | :--- |
| `holiday` | Categorical | US national holiday or MN State Fair label; `NaN` on regular days |
| `temp` | Numeric (K) | Atmospheric temperature in Kelvin |
| `rain_1h` | Numeric (mm) | Millimetres of rainfall in the past hour |
| `snow_1h` | Numeric (mm) | Millimetres of snowfall in the past hour |
| `clouds_all` | Numeric (%) | Percentage of cloud coverage |
| `weather_main` | Categorical | Short weather description (e.g., Clear, Clouds, Rain) |
| `weather_description` | Categorical | Verbose weather description (excluded — redundant with `weather_main`) |
| `date_time` | Datetime | Hourly timestamp of the observation |
| `traffic_volume` | Numeric | **Target:** Westbound hourly vehicle count (0 to ~7,280) |

### Data Cleaning

Two quality issues were identified and corrected before any modeling:

1. **Duplicate Hourly Records:** The raw file contains 48,204 rows but multiple duplicate `date_time` entries exist (same hour logged more than once). All duplicates were removed using a keep-first strategy on `date_time`, reducing the dataset to 40,575 unique hourly records.

2. **Impossible Temperature Values:** 10 rows contain `temp = 0 K` (-273.15 C), which is a physical impossibility representing sensor failures. These rows were removed. Final clean dataset: **40,565 rows**.

### Train/Test Split

| Partition | Rows | Percentage |
| :--- | :---: | :---: |
| Training Set (`X_train`) | 32,452 | 80% |
| Test Set (`X_test`) | 8,113 | 20% |

Split parameters: `test_size=0.2`, `random_state=42`. The same fixed split is used across all five models to ensure a fair comparison.

---

## 3. Feature Engineering

Six additional features were derived from the raw columns before model training:

| Feature | Derivation | Purpose |
| :--- | :--- | :--- |
| `is_holiday` | 1 if `holiday` is not NaN, else 0 | Binary indicator; replaces the text holiday column |
| `hour` | `date_time.dt.hour` | Raw hour of day (0--23) |
| `day_of_week` | `date_time.dt.dayofweek` | Day index (0 = Monday, 6 = Sunday) |
| `month` | `date_time.dt.month` | Month index (1--12) |
| `is_weekend` | 1 if `day_of_week` in {5, 6}, else 0 | Binary weekend indicator |
| `hour_sin` | sin(2 * pi * hour / 24) | Cyclical hour encoding — sine component |
| `hour_cos` | cos(2 * pi * hour / 24) | Cyclical hour encoding — cosine component |

**Why cyclical encoding for hour?**
A raw integer 0--23 creates an artificial discontinuity: hour 23 and hour 0 appear maximally distant despite being one hour apart. The sine/cosine transformation maps the 24-hour cycle onto a unit circle, ensuring continuity at midnight.

**Final feature set used for modeling:**

- Numeric: `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `hour`, `day_of_week`, `month`, `is_holiday`, `is_weekend`, `hour_sin`, `hour_cos`
- Categorical (one-hot encoded): `weather_main`

---

## 4. Pipeline Architecture

> [!IMPORTANT]
> All preprocessing transformers are fitted **exclusively on `X_train`**. The held-out test set `X_test` is only transformed (never fitted), guaranteeing zero data leakage.

```mermaid
graph TD
    A[Raw CSV: data/Metro_Interstate_Traffic_Volume.csv] --> B[Deduplication on date_time]
    B --> C[Remove temp = 0K rows]
    C --> D[Feature Engineering: is_holiday, cyclical hour, temporal features]
    D --> E[80:20 Train/Test Split, random_state=42]
    E --> F[X_train: 32,452 samples]
    E --> G[X_test: 8,113 samples]

    subgraph Leakage-Safe sklearn Pipeline
        F --> H[ColumnTransformer: fit_transform on X_train only]
        H --> I[StandardScaler: 11 numeric features]
        H --> J[OneHotEncoder: weather_main]
        I --> K[Estimator: DT / RF / GBR / SVR / KNN]
        J --> K
    end

    G --> L[ColumnTransformer: transform only]
    K --> M[Fitted Model]
    M --> N[Evaluate on X_test]
    L --> N
```

---

## 5. Algorithms Implemented

Five non-linear regression algorithms are implemented and evaluated in `notebooks/regression.ipynb`:

### Decision Tree Regressor

Recursively partitions feature space by selecting the feature and threshold that minimizes weighted child variance (MSE-equivalent split criterion). Depth and minimum sample constraints control overfitting.

Key hyperparameters tuned: `max_depth`, `min_samples_split`, `min_samples_leaf`.

### Random Forest Regressor

Bootstrap ensemble of decision trees. Each tree is trained on a random sample (with replacement) of the training data and a random subset of features at each split. Final prediction is the arithmetic mean across all trees.

Key hyperparameters tuned: `n_estimators`, `max_depth`, `max_features`.

### Gradient Boosting Regressor

Sequentially fits shallow trees to the residual errors of the current ensemble. Each new tree corrects the mistakes of all previous trees. The learning rate (shrinkage) controls the step size in function space.

Key hyperparameters tuned: `n_estimators`, `learning_rate`, `max_depth`.

### Support Vector Regressor (SVR)

Fits a hyperplane within an epsilon-insensitive tube around the training data using a radial basis function (RBF) kernel. Predictions are made by support vectors — training points outside the tube. Sensitive to feature scale; requires `StandardScaler`.

Key hyperparameters tuned: `C`, `epsilon`, `gamma` (via RBF kernel).

### K-Nearest Neighbors Regressor (KNN)

Predicts by averaging the target values of the k training points closest to the query point under Euclidean distance. Non-parametric; no explicit training phase. Sensitive to feature scale; requires `StandardScaler`.

Key hyperparameter tuned: `n_neighbors`.

---

## 6. Results and Benchmarks

All five models are evaluated on the held-out test set of **8,113 samples**, ranked by test R2 score descending. The top 2 models also include 5-fold cross-validation.

| Rank | Algorithm | Test R2 | RMSE (veh/hr) | MAE (veh/hr) | 5-Fold CV R2 | Key Hyperparameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **Gradient Boosting Regressor** | **0.9460** | **459.26** | **275.48** | 0.9437 +/- 0.0019 | `lr=0.2`, `n_estimators=200`, `max_depth=4` |
| 2 | **Random Forest Regressor** | **0.9454** | **461.74** | **266.93** | 0.9413 +/- 0.0016 | `n_estimators=100`, `max_depth=10` |
| 3 | Decision Tree Regressor | 0.9396 | 485.70 | 284.42 | — | `max_depth=8` |
| 4 | K-Nearest Neighbors (KNN) | 0.9286 | 528.02 | 338.88 | — | `n_neighbors=9` |
| 5 | Support Vector Regressor | 0.8261 | 824.15 | 555.43 | — | `kernel='rbf'`, `C=50.0` |

### Key Analytical Observations

**Why ensemble methods lead:** Gradient Boosting and Random Forest both capture the bimodal, non-linear 24-hour traffic distribution (morning and evening commuter peaks) without manual feature engineering for peak periods. Bagging and boosting independently reduce prediction variance, the dominant error source in hourly traffic data.

**SVR performance:** The default `C=1.0` severely over-regularizes SVR, yielding a baseline R2 of only 0.18. Tuning to `C=50.0` recovers an R2 of 0.8261 (+0.6447 improvement), demonstrating how critical regularization strength is for SVR on large-scale datasets.

**KNN with scaling:** Without `StandardScaler`, `temp` in Kelvin (range ~250--310) dominates Euclidean distances over binary features. Standardizing all features allows KNN to find genuinely meaningful neighbors.

**Feature importances:** Across all tree-based models, `hour`, `hour_sin`, `hour_cos`, and `day_of_week` account for the majority of variance reduction, confirming that daily commuter timing is the primary driver of traffic volume — not weather conditions.

---

## 3. Clustering Track (Review 2)

### 3.1 Problem Statement & Dataset
The **Review 2 Clustering Track** implements an unsupervised customer segmentation pipeline on the **UCI Online Retail II** dataset (`mashlyn/online-retail-ii-uci`), ingested directly via the official `kagglehub` API. Spanning two full calendar years (2009–2011) of real-world UK e-commerce transactions, the dataset enables behavioral segmentation to tailor commercial marketing strategies.

| Field | Value |
| :--- | :--- |
| **Dataset Source** | UCI Machine Learning Repository / Kaggle (`mashlyn/online-retail-ii-uci`) |
| **Raw Ingestion** | `kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, 'mashlyn/online-retail-ii-uci', 'online_retail_II.csv')` |
| **Raw Dimensions** | 1,067,371 rows × 8 columns |
| **Cleaned Line Items** | 805,620 validated purchase records (after pruning cancellations and missing IDs) |
| **Customer Entity Space** | **5,878 unique customer accounts** |
| **Notebook** | [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) |

### 3.2 Preprocessing & RFM Feature Engineering
1. **Cleaning Protocol**:
   - **Missing Customer IDs**: 243,007 guest checkout rows were isolated, preserving 824,364 customer-assigned transactions.
   - **Transaction Reversals**: Filtered out cancellation invoices prefixed with `'C'`.
   - **Value Sanity**: Retained strictly valid purchases with `Quantity > 0` and `Price > 0`.
2. **RFM Behavioral Framework**:
   - **Recency ($R$)**: Days between most recent customer order and analysis cutoff ($T_{ref} = \max(\text{InvoiceDate}) + 1\text{ day}$).
   - **Frequency ($F$)**: Count of distinct completed purchase invoices per customer.
   - **Monetary ($M$)**: Total cumulative spending across all transactions ($\sum \text{Quantity} \times \text{Price}$).
   - **Average Order Value ($AOV = M / F$)**: Expenditure efficiency per transaction cart.
3. **Distribution Normalization**:
   - Severe right-skewness (skewness $> 15$) was normalized using natural log transformation ($\log(1 + x)$ via `np.log1p`).
   - Features were standardized with `StandardScaler` ($\mu = 0, \sigma = 1$), ensuring equal geometric weighting in Euclidean distance calculations.

### 3.3 Algorithms & Hyperparameter Selection
- **Algorithm 1 — K-Means Clustering**:
  - Evaluated across $k \in [2, 8]$ using the **Elbow Curve (Inertia/WCSS)** and **Silhouette Score curve**.
  - Selected optimal **$k = 4$** based on the elbow bend point, a local Silhouette peak ($0.2813$), and four highly actionable customer personas.
- **Algorithm 2 — Agglomerative Hierarchical Clustering**:
  - Compared linkage strategies using the **Cophenetic Correlation Coefficient**: `ward` ($0.5161$), `complete` ($0.4950$), `average` ($0.6854$), `single` ($0.4735$).
  - Constructed a truncated **Dendrogram** ($p=30$) with Ward linkage, identifying a natural cut threshold at distance $68.0$ intersecting **4 distinct branches**.

---

## 4. Clustering Benchmarks and Personas

### 4.1 Consolidated Metric Comparison (Review 2 Rubric B2)

| Algorithm | Optimal $k$ | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Calinski-Harabasz Index ($\uparrow$) | Compute Complexity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means Clustering (Selected)** | **4** | **0.2813** | **1.1411** | **3,161.1** | **$O(K \cdot N \cdot d)$ (~0.05 s)** |
| **Agglomerative Hierarchical (Ward)** | 4 | 0.2134 | 1.3609 | 2,593.3 | $O(N^2 \log N)$ (~1.42 s) |

> **Key Takeaway**: K-Means outperforms Agglomerative Hierarchical clustering across every metric, demonstrating tighter intra-cluster cohesion (Silhouette $0.2813$), lower cluster overlap (DBI $1.1411$), and superior between-cluster variance ratios (CHI $3,161.1$).

### 4.2 Dimensionality Reduction & Visualizations (Rubric B3)
- **PCA 2D Projection**: Projects the 4D standardized feature space onto the two primary eigenvectors, capturing **84.3% of cumulative variance** (PC1: 65.1%, PC2: 19.2%). Centroids and decision boundaries are clearly separated.
- **t-SNE 2D Manifold**: Non-linear manifold embedding ($\text{perplexity}=30$) reveals isolated, dense clusters with clean topological corridors, proving that customer segments reflect genuine behavioral boundaries.

### 4.3 Actionable Customer Persona Matrix

| Cluster ID | Persona Archetype | Customer Share | Behavioral Profile | Targeted Marketing Action |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Champions & VIP High-Rollers** | ~17.5% | Recency ~10 days, Frequency ~15 orders, Monetary £5,000+ | Dedicated VIP account manager, early line access, wholesale tiered volume rebates. |
| **2** | **Loyal Consistent Regulars** | ~33.8% | Recency ~45 days, Frequency ~5 orders, Monetary ~£1,500 | Cross-selling recommendations, free shipping thresholds to increase $AOV$, automated replenishment. |
| **3** | **At-Risk / Lapsing Customers** | ~24.1% | Recency ~200 days, Frequency ~2 orders, Monetary ~£600 | Win-back discount incentives ("15% off next order"), churn feedback surveys. |
| **0** | **Lost / Dormant Low-Spenders** | ~24.6% | Recency ~450+ days, Frequency = 1 order, Monetary ~£200 | Low-cost clearance email re-targeting; suppress physical mailers to preserve marketing margin. |

---

## 5. Repository Structure

```text
ml_capstone/
|
├── README.md                               # Project documentation and benchmarks
├── requirements.txt                        # Version-pinned Python dependencies (including kagglehub, scipy)
├── .gitignore
|
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv # Regression dataset (40,565 clean rows)
|
├── models/                                 # Serialized production model artifacts
│   ├── kmeans_customer_segmentation.pkl    # Trained K-Means clustering model (k=4)
│   └── scaler_customer_segmentation.pkl    # Standard scaler pipeline artifact
|
└── notebooks/
    ├── regression.ipynb                    # Full 5-model regression pipeline (Review 1)
    └── clustering.ipynb                    # End-to-end customer segmentation pipeline (Review 2)
```

---

## 6. Installation and Quick Start


### Prerequisites

- Python 3.10 or higher (tested on 3.12)
- pip 23+

### Step 1 — Clone and Enter the Repository

```bash
git clone https://github.com/Dakshin10/ml_capstone.git
cd ml_capstone
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

| Package | Minimum Version | Purpose |
| :--- | :---: | :--- |
| `numpy` | 1.24.0 | Numerical computation |
| `pandas` | 2.0.0 | Data loading and feature engineering |
| `matplotlib` | 3.7.0 | Static plots and charts |
| `seaborn` | 0.12.0 | Statistical visualization |
| `scikit-learn` | 1.2.0 | Preprocessing, models, metrics |
| `scipy` | 1.10.0 | Hierarchical clustering & cophenetic correlation |
| `kagglehub` | 0.2.0 | Dataset ingestion directly from Kaggle |
| `joblib` | 1.2.0 | Model serialization |
| `jupyter` | 1.0.0 | Notebook server |
| `ipykernel` | 6.0.0 | Python kernel for VS Code / JupyterLab |

> [!NOTE]
> scikit-learn 1.2+ is required. Earlier versions use the deprecated `OneHotEncoder(sparse=False)` parameter; this codebase uses `sparse_output=False`.

### Step 4 — Open and Run the Notebooks

**Jupyter:**
```bash
jupyter notebook
```
- For **Regression (Review 1)**: Navigate to [`notebooks/regression.ipynb`](notebooks/regression.ipynb) and select **Kernel -> Restart and Run All**.
- For **Clustering (Review 2)**: Navigate to [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) and select **Kernel -> Restart and Run All**.

**VS Code:**
1. Open [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) or [`notebooks/regression.ipynb`](notebooks/regression.ipynb)
2. Click the kernel selector (top-right corner)
3. Select the Python interpreter from your active environment
4. Click **Run All**

---

## 7. Reproducibility

All stochastic operations strictly enforce `random_state=42`:

- `KMeans(random_state=42)` — deterministic initial centroid placement (`k-means++`)
- `TSNE(random_state=42)` — deterministic 2D manifold projection
- `PCA(random_state=42)` — deterministic principal component extraction
- `train_test_split(random_state=42)` — identical partitions across runs

Running [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) from top to bottom reproduces the exact metric values, dendrogram branches, and cluster persona assignments.

---

<div align="center">

B.Tech Computer Science and Engineering &nbsp;|&nbsp; 23CSE301 Machine Learning Capstone &nbsp;|&nbsp; Review 1 & Review 2

</div>

