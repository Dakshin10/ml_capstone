<div align="center">

# 23CSE301 — Machine Learning Capstone | Review 2

**Unsupervised Customer Segmentation on Online Retail II using K-Means & Agglomerative Hierarchical Clustering**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Review%202%20Clustering%20Ready-brightgreen.svg)]()

[Overview](#1-project-overview) &nbsp;|&nbsp;
[Dataset & Ingestion](#2-dataset-and-ingestion) &nbsp;|&nbsp;
[Preprocessing & RFM](#3-preprocessing-and-rfm-feature-engineering) &nbsp;|&nbsp;
[Clustering Algorithms](#4-clustering-algorithms) &nbsp;|&nbsp;
[Benchmark Results](#5-consolidated-benchmark-results) &nbsp;|&nbsp;
[Visualizations](#6-dimensionality-reduction-and-cluster-visualizations) &nbsp;|&nbsp;
[Personas](#7-customer-personas-and-business-strategy) &nbsp;|&nbsp;
[Repository Structure](#8-repository-structure) &nbsp;|&nbsp;
[Quick Start](#9-installation-and-quick-start)

---

</div>

## 1. Project Overview

This branch (`review2_clustering`) delivers the complete, self-contained **Clustering Track for Review 2** of the **23CSE301 Machine Learning Capstone**.

The objective is to perform unsupervised customer segmentation on real-world retail transactions from the **UCI Online Retail II** dataset. Raw invoice transactions are aggregated into high-fidelity behavioral customer profiles using the **RFM (Recency, Frequency, Monetary)** framework, normalized, and partitioned using two distinct unsupervised clustering paradigms:
1. **Centroid-based Partitioning**: K-Means Clustering with Lloyd's optimization.
2. **Connectivity-based Hierarchical Grouping**: Agglomerative Hierarchical Clustering with Ward's minimum-variance linkage.

### Scope Summary
| Property | Value |
| :--- | :--- |
| **Track** | Review 2 — Clustering Track (Unsupervised Learning) |
| **Dataset** | UCI Online Retail II (`mashlyn/online-retail-ii-uci` via `kagglehub`) |
| **Entity for Clustering** | Unique Commercial Customers (5,878 active accounts) |
| **Algorithms Evaluated** | K-Means Clustering, Agglomerative Hierarchical Clustering |
| **Evaluation Metrics** | Silhouette Score, Davies-Bouldin Index (DBI), Calinski-Harabasz Index (CHI) |
| **Dimensionality Reduction** | Principal Component Analysis (PCA 2D), t-SNE 2D Manifold Embedding |
| **Notebook Deliverable** | [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) (52 cells, executed top-to-bottom with zero errors) |
| **Serialized Artifacts** | [`models/kmeans_customer_segmentation.pkl`](models/kmeans_customer_segmentation.pkl), [`models/scaler_customer_segmentation.pkl`](models/scaler_customer_segmentation.pkl) |

---

## 2. Dataset and Ingestion

The dataset spans two calendar years (01/12/2009 to 09/12/2011) of real transactional logs for a UK-based non-store online retail company specializing in unique all-occasion giftware.

### Ingestion via KaggleHub
The dataset is fetched dynamically using the official `kagglehub` API with the Pandas adapter as specified in the project guidelines:

```python
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "online_retail_II.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "mashlyn/online-retail-ii-uci",
  file_path,
  pandas_kwargs={"compression": "zip", "encoding": "unicode_escape"}
)

print("First 5 records:", df.head())
```

### Raw Attributes
| Column | Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `Invoice` | object | Identifier | 6-digit transaction number; prefix 'C' denotes cancellation |
| `StockCode` | object | Product ID | 5-digit integral code uniquely identifying each product |
| `Description` | object | Metadata | Name and textual description of the product |
| `Quantity` | int64 | Numerical | Units of product purchased per line item |
| `InvoiceDate` | object | Temporal | Timestamp of the transaction |
| `Price` | float64 | Numerical | Unit price in British Pounds Sterling (£) |
| `Customer ID` | float64 | Entity Key | 5-digit integral identifier assigned to registered customers |
| `Country` | object | Categorical | Country where the customer resides |

---

## 3. Preprocessing and RFM Feature Engineering

### 3.1 Data Cleaning Protocol
1. **Customer ID Isolation**: Exactly 243,007 records (~22.77%) lack a `Customer ID` (anonymous guest checkouts). Because customer segmentation clusters identifiable customer accounts, guest purchases are filtered out, preserving 824,364 customer-assigned line items.
2. **Cancellation Pruning**: Invoices prefixed with `'C'` represent order cancellations and return adjustments. These are filtered to ensure frequency and spend reflect actual completed purchases.
3. **Transaction Sanity**: Records with non-positive quantities (`Quantity <= 0`) or non-positive unit prices (`Price <= 0`) correspond to ledger adjustments, damages, or discounts; these are removed.
4. **Deduplication**: Exact duplicate records are pruned.
5. **Cleaned Volume**: **805,620 high-quality commercial line items** representing **5,878 unique customer accounts** are retained.

### 3.2 RFM Behavioral Framework
Because individual invoice rows represent line items, we aggregate transactions to the customer level relative to a reference cutoff date ($T_{ref} = \max(\text{InvoiceDate}) + 1\text{ day}$):
- **Recency ($R$)**: Days between the customer's most recent order and $T_{ref}$. Low recency signals active brand affinity.
- **Frequency ($F$)**: Count of distinct completed purchase invoices. High frequency indicates habitual customer loyalty.
- **Monetary ($M$)**: Cumulative gross spending across all invoices ($\sum \text{Quantity} \times \text{Price}$). Identifies high-value revenue drivers.
- **Average Order Value ($AOV = M / F$)**: Engineered metric quantifying basket size expenditure efficiency.
- **Tenure ($T$)**: Customer relationship lifespan in days ($\max(\text{Date}) - \min(\text{Date})$).

### 3.3 Distribution Normalization & Scaling
- **Log Transformation**: Financial and behavioral transaction metrics exhibit severe positive right-skewness (skewness $> 15$) following power-law distributions. We apply natural log transformation ($\log(1 + x)$ via `np.log1p`) to compress long tails and stabilize variance.
- **Standardization**: Features are scaled using `StandardScaler` ($\mu = 0, \sigma = 1$). This eliminates scale disparities, ensuring all dimensions contribute proportionally in Euclidean distance space.

---

## 4. Clustering Algorithms

### 4.1 Algorithm 1: K-Means Clustering
- **Theory**: Iteratively partitions observations into $k$ spherical clusters by minimizing the within-cluster sum of squares (WCSS / Inertia):
  $$J = \sum_{j=1}^K \sum_{x_i \in S_j} \|x_i - \mu_j\|^2$$
- **Optimal $k$ Selection**:
  - Swept $k \in [2, 8]$ computing Inertia, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
  - **Elbow Curve**: A distinct inflection point is identified at **$k = 4$** ($\text{Inertia} \approx 8,993$), after which within-cluster variance reduction exhibits diminishing returns.
  - **Silhouette Curve**: Confirms a strong local peak at $k=4$ ($0.2813$) with balanced, highly differentiable cluster boundaries.

### 4.2 Algorithm 2: Agglomerative Hierarchical Clustering
- **Theory**: Deterministic bottom-up hierarchical clustering where each observation starts in its own singleton cluster and merges iteratively based on a dissimilarity metric.
- **Linkage Strategy Benchmark**: Evaluated using the **Cophenetic Correlation Coefficient**:
  - **Ward's Linkage**: $0.5161$ (Minimizes total within-cluster variance increase; creates compact, spherical, balanced clusters).
  - **Complete Linkage**: $0.4950$ (Max pairwise distance).
  - **Average Linkage**: $0.6854$ (Average pairwise distance; prone to single-point chaining in practice).
  - **Single Linkage**: $0.4735$ (Minimum pairwise distance; severe chaining).
- **Dendrogram**: Constructed with Ward's linkage (top 30 merges). A horizontal cut at distance threshold $68.0$ cleanly intersects **4 primary branches**, confirming the optimal $k=4$ partition.

---

## 5. Consolidated Benchmark Results

Per Review 2 Rubric B2, all algorithms are quantitatively compared on the same preprocessed feature space:

| Algorithm | Optimal $k$ | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Calinski-Harabasz Index ($\uparrow$) | Compute Complexity | Convergence Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means Clustering (Selected)** | **4** | **0.2813** | **1.1411** | **3,161.1** | **$O(K \cdot N \cdot d)$** | **~0.05 s** |
| **Agglomerative Hierarchical (Ward)** | 4 | 0.2134 | 1.3609 | 2,593.3 | $O(N^2 \log N)$ | ~1.42 s |

### Key Analytical Takeaways:
1. **Compactness & Separation**: K-Means achieves a higher Silhouette score ($0.2813$ vs. $0.2134$), confirming superior cluster cohesion and distinctness.
2. **Cluster Overlap**: K-Means yields a lower Davies-Bouldin Index ($1.1411$ vs. $1.3609$), reflecting cleaner boundary margins between segments.
3. **Variance Dispersion**: K-Means achieves a higher Calinski-Harabasz score ($3,161.1$ vs. $2,593.3$), verifying that the ratio of between-cluster to within-cluster scatter is maximized.
4. **Computational Scalability**: K-Means converges in $0.05$ seconds with linear $O(N)$ scalability, making it the production choice for real-time customer re-clustering.

---

## 6. Dimensionality Reduction and Cluster Visualizations

Per Review 2 Rubric B3, the 4-dimensional standardized feature space is projected into 2D to validate topological separation:

### 6.1 Principal Component Analysis (PCA 2D)
- **Variance Explained**: PC1 captures **65.1%** of variance; PC2 captures **19.2%** of variance (**84.3% cumulative variance explained**).
- 2D scatter plots for both algorithms show distinct spatial regions, with K-Means centroids positioned centrally in their respective cluster densities.

### 6.2 t-Distributed Stochastic Neighbor Embedding (t-SNE 2D)
- Non-linear manifold learning with $\text{perplexity} = 30$ and 1,000 iterations.
- Visualizes non-linear local neighborhood topologies; reveals tightly bounded, isolated islands of customer archetypes with wide empty corridors, proving that the clusters represent genuine behavioral groupings.

---

## 7. Customer Personas and Business Strategy

By inverse-transforming cluster centroids back to raw business units (£, order counts, days), we derive four actionable customer personas:

| Cluster ID | Persona Archetype | Customer Share | Median Profile | Targeted Business & Marketing Strategy |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Champions & VIP High-Rollers** | ~17.5% | Recency: 10 days<br>Frequency: 15 orders<br>Monetary: £5,000+ | **VIP Loyalty Rewards**: Dedicated account manager, early preview access to new product catalogs, tiered wholesale volume rebates. |
| **2** | **Loyal Consistent Regulars** | ~33.8% | Recency: 45 days<br>Frequency: 5 orders<br>Monetary: £1,500 | **Upselling & Basket Growth**: Cross-sell personalized product bundles, free-shipping minimum thresholds to grow $AOV$, automated re-ordering. |
| **3** | **At-Risk / Lapsing Customers** | ~24.1% | Recency: 200 days<br>Frequency: 2 orders<br>Monetary: £600 | **Win-Back Campaigns**: Time-sensitive re-activation incentives ("We miss you — 15% off next order"), churn feedback surveys. |
| **0** | **Lost / Dormant Low-Spenders** | ~24.6% | Recency: 450+ days<br>Frequency: 1 order<br>Monetary: £200 | **Low-Cost Clearance Re-targeting**: Automated clearance email blasts; suppress expensive direct mail to conserve operational marketing spend. |

---

## 8. Repository Structure

```text
ml_capstone/
|
├── README.md                               # Complete project documentation and benchmarks
├── requirements.txt                        # Version-pinned Python dependencies
├── .gitignore
|
├── data/
│   └── download_dataset.py                 # Automated KaggleHub ingestion script
|
├── models/                                 # Serialized production model artifacts
│   ├── kmeans_customer_segmentation.pkl    # Trained K-Means clustering model (k=4)
│   └── scaler_customer_segmentation.pkl    # Standard scaler pipeline artifact
|
└── notebooks/
    └── clustering.ipynb                    # Complete Review 2 Clustering pipeline
                                            # (EDA, Preprocessing, RFM, K-Means,
                                            #  Hierarchical, Metrics, PCA, t-SNE)
```

---

## 9. Installation and Quick Start

### Step 1 — Clone and Enter the Repository
```bash
git clone https://github.com/Dakshin10/ml_capstone.git
cd ml_capstone
git checkout review2_clustering
```

### Step 2 — Create and Activate a Virtual Environment
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

### Step 4 — Run the Automated Dataset Ingestion Script (Optional)
```bash
python data/download_dataset.py
```

### Step 5 — Open and Execute the Notebook
```bash
jupyter notebook
```
Navigate to [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) and click **Kernel -> Restart and Run All**.

---

## 10. Reproducibility Guarantee

All stochastic operations strictly enforce `random_state=42`:
- `KMeans(random_state=42)` — deterministic initial centroid placement (`k-means++`)
- `TSNE(random_state=42)` — deterministic 2D manifold projection
- `PCA(random_state=42)` — deterministic principal component extraction

Executing [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) top-to-bottom reproduces the exact metric values, dendrogram branches, and cluster persona assignments.

---

<div align="center">

B.Tech Computer Science and Engineering &nbsp;|&nbsp; 23CSE301 Machine Learning Capstone &nbsp;|&nbsp; Review 2

</div>
