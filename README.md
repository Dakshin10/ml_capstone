# 23CSE301 — Machine Learning Capstone | Complete Repository (Review 1 & Review 2)

An end-to-end Machine Learning pipeline encompassing **Regression**, **Classification**, and **Clustering** tracks for the B.Tech. CSE 23CSE301 Machine Learning Capstone project.

---

## 1. Project Overview & Problem Tracks

| Track | Dataset | Problem Statement | Target Variable | Type | Evaluated Algorithms | Primary Benchmark Metric |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Regression** | `Metro_Interstate_Traffic_Volume.csv` | Predict hourly eastbound traffic volume on I-94 | `traffic_volume` | Continuous ($[0, 7280]$ veh/hr) | **10 Models** (Linear, Ridge, Lasso, ElasticNet, Poly, Decision Tree, Random Forest, GBR, SVR, KNN) | $R^2$, RMSE, MAE, 5-Fold CV $R^2$ |
| **Classification** | `bank-full.csv` | Predict term deposit subscription from telemarketing data | `y` (`yes` / `no`) | Binary ($0 / 1$, 88.3% vs. 11.7%) | **10 Models** (Part A: Logistic, KNN, Naive Bayes, Decision Tree, SVC; Part B: Random Forest, AdaBoost, GBC, Bagging, MLP) | Weighted F1, ROC-AUC, Accuracy, Confusion Matrix |
| **Clustering** | `online_retail_II.csv` | Unsupervised RFM customer segmentation for e-commerce | Customer Segments (Persona Matrix) | Unsupervised ($k=4$) | **2 Algorithms** (K-Means with Elbow & Silhouette, Agglomerative Hierarchical with Dendrogram & Linkage) | Silhouette Score, Davies-Bouldin, Calinski-Harabasz |

---

## 2. Track Architecture & Implementation Details

### Track 1: Hourly Interstate Traffic Volume Prediction (`notebooks/regression.ipynb`)
- **Dataset:** 48,204 records from the UCI Machine Learning Repository.
- **Data Cleaning:** Removal of exact duplicates and physical sensor anomalies (0 Kelvin temperature dropouts, rainfall spikes $> 500$ mm).
- **Feature Engineering:**
  - Temporal features: `hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday`.
  - Continuous periodic trigonometric representations: $\text{hour}_{\sin} = \sin(2\pi \cdot \text{hour} / 24)$ and $\text{hour}_{\cos} = \cos(2\pi \cdot \text{hour} / 24)$.
  - Celsius conversion: $\text{temp\_celsius} = \text{temp} - 273.15$.
- **Preprocessing Pipeline:** `ColumnTransformer` with `StandardScaler` on numerical features and `OneHotEncoder(handle_unknown='ignore')` on categorical weather categories, fitted strictly on the 80% training partition.
- **All 10 Evaluated Algorithms:**
  1. *Linear Regression* (Ordinary Least Squares baseline)
  2. *Ridge Regression* (L2 regularization with cross-validated $\alpha$ tuning)
  3. *Lasso Regression* (L1 regularization with feature sparsity analysis)
  4. *ElasticNet Regression* (Hybrid L1 + L2 penalty)
  5. *Polynomial Regression* (Degree 2 feature interactions)
  6. *Decision Tree Regressor* (Depth-tuned tree partitioning)
  7. *Random Forest Regressor* (150 bagged de-correlated trees)
  8. *Gradient Boosting Regressor* (Sequential gradient boosting, learning rate tuning)
  9. *Support Vector Regressor (SVR)* (RBF kernel with $C=50.0$)
  10. *K-Nearest Neighbors Regressor (KNN)* (Distance-weighted local regression, $k=9$)
- **Top Performer:** Gradient Boosting Regressor ($R^2 \approx 0.9529$, RMSE $\approx 430.65$ veh/hr, 5-Fold CV $R^2 = 0.9485 \pm 0.0020$).

---

### Track 2: Bank Marketing Term Deposit Prediction (`notebooks/classification.ipynb`)
- **Dataset:** 45,211 instances from UCI Bank Marketing dataset (`bank-full.csv`).
- **Critical Leakage Guard:** `duration` is **strictly excluded** from all input predictors prior to modeling, as call duration is unknown before a call is conducted and artificially inflates metrics.
- **Handling Imbalance & Preprocessing:** Stratified 80/20 train/test split preserving the 88.3% vs. 11.7% class ratio. One-hot encoding for categorical variables and standard scaling for numerical features.
- **All 10 Evaluated Algorithms:**
  - *Part A (Review 1):*
    1. **Logistic Regression** (Linear baseline, odds ratio interpretation)
    2. **K-Nearest Neighbors (KNN)** (Distance-based classification, $k=5$)
    3. **Gaussian Naive Bayes** (Probabilistic classifier, conditional independence assumption)
    4. **Decision Tree Classifier** (Tree partitioning, Gini importance analysis)
    5. **Support Vector Classifier (SVC)** (RBF kernel, maximal margin separation)
  - *Part B (Review 2):*
    6. **Random Forest Classifier** (150 trees, max_depth=12, ensemble bagging)
    7. **AdaBoost Classifier** (100 decision stumps, adaptive boosting)
    8. **Gradient Boosting Classifier** (150 estimators, stage-wise error reduction)
    9. **Bagging Classifier** (50 bagged decision trees)
    10. **MLP Classifier** (Multi-Layer Perceptron neural network, hidden layers: 64, 32)
- **Top Performer:** Gradient Boosting & Random Forest (ROC-AUC $\approx 0.79$, Weighted F1 $\approx 0.88$).

---

### Track 3: Unsupervised Retail Customer Segmentation (`notebooks/clustering.ipynb`)
- **Dataset:** 525,461 transaction records from the Online Retail II UCI dataset (`data/online_retail_II.csv`).
- **Data Cleaning:** Removal of missing CustomerIDs, cancellation invoices (prefixed with 'C'), negative quantities, and zero unit prices.
- **RFM Feature Engineering:**
  - **Recency:** Days elapsed since last customer purchase.
  - **Frequency:** Total number of distinct purchases/invoices.
  - **Monetary:** Total monetary spend ($Quantity \times Price$).
- **Data Transformation:** Right-skew log-transformation ($\log_{1p}$) followed by `StandardScaler` standardization.
- **Algorithms & Visualizations:**
  1. **K-Means Clustering:**
     - Elbow Method (Inertia vs. $k$) and Silhouette Score analysis for $k \in [2, 8]$.
     - Optimal clusters: $k=4$.
  2. **Agglomerative Hierarchical Clustering:**
     - Full Dendrogram visualization with linkage comparison (Ward, Complete, Average).
  3. **Dimensionality Reduction:**
     - 2D & 3D Principal Component Analysis (PCA) cluster projections.
     - 2D t-SNE non-linear cluster visualization.
  4. **Validation Metrics:** Silhouette Score ($\approx 0.38$), Davies-Bouldin Index ($\approx 0.94$), Calinski-Harabasz Index ($\approx 2600+$).
  5. **Persona Profiling Matrix:** Champions / High-Value, Loyal Core, Promising / Recent Buyers, and At-Risk / Hibernating Customers.
  6. **Model Artifacts:** Saved to `models/kmeans_customer_segmentation.pkl` and `models/scaler_customer_segmentation.pkl`.

---

## 3. Repository Structure

```text
ml_capstone/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── bank-full.csv                      # Classification: Bank Marketing dataset
│   ├── Metro_Interstate_Traffic_Volume.csv # Regression: Interstate traffic volume dataset
│   ├── online_retail_II.csv               # Clustering: Online Retail II dataset
│   ├── insurance.csv                      # Regression: Medical charges dataset (alternative)
│   ├── download_dataset.py                # Dataset download utility script
│   └── *.png                              # Exported high-resolution publication charts
├── models/
│   ├── .gitkeep
│   ├── kmeans_customer_segmentation.pkl   # Serialized fitted K-Means clustering model
│   └── scaler_customer_segmentation.pkl   # Serialized fitted RFM StandardScaler
└── notebooks/
    ├── regression.ipynb                   # Unified 10-Algorithm Regression Pipeline
    ├── classification.ipynb               # Unified 10-Algorithm Classification Pipeline
    └── clustering.ipynb                   # Unsupervised RFM Clustering Pipeline
```

---

## 4. Installation & Environment Setup

### Prerequisites
Python 3.10+ (tested on Python 3.12 and Python 3.14).

```bash
# Clone the repository
git clone https://github.com/SH-Nihil-Mukkesh-25/ml_capstone.git
cd ml_capstone

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Running the Notebooks
Launch Jupyter Lab or Notebook:
```bash
jupyter lab
```
Run all cells in sequence:
- `notebooks/regression.ipynb`
- `notebooks/classification.ipynb`
- `notebooks/clustering.ipynb`

All notebooks include dynamic path resolution and will execute without errors whether launched from the workspace root or directly inside `notebooks/`.

---

## 5. Consolidated Benchmark Results

### Regression Track (10 Models) — Target: `traffic_volume`
| Rank | Algorithm | Test $R^2$ | RMSE (veh/hr) | MAE (veh/hr) | 5-Fold CV $R^2$ | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Gradient Boosting Regressor** | **0.9529** | **430.65** | **255.57** | **0.9485 ± 0.0020** | Top performer; captures cyclical peak hours |
| **2** | **Random Forest Regressor** | **0.9492** | **447.50** | **257.26** | **0.9454 ± 0.0024** | Robust ensemble; low variance |
| **3** | Decision Tree Regressor | 0.9372 | 497.33 | 291.74 | N/A | Tuned max_depth=8 |
| **4** | K-Nearest Neighbors (KNN) | 0.9293 | 527.55 | 332.08 | N/A | Distance-weighted local regression ($k=9$) |
| **5** | Support Vector Regressor (SVR) | 0.8430 | 786.44 | 530.07 | N/A | RBF kernel ($C=50.0$) |
| **6** | Polynomial Regression (deg=2) | 0.8419 | 789.05 | 602.42 | N/A | Degree-2 non-linear feature interactions |
| **7** | Ridge Regression | 0.7176 | 1054.72 | 820.09 | N/A | L2 regularization ($\alpha \approx 48.33$) |
| **8** | Linear Regression | 0.7176 | 1054.69 | 820.04 | N/A | Baseline OLS formulation |
| **9** | Lasso Regression | 0.7176 | 1054.70 | 820.07 | N/A | L1 regularization ($\alpha \approx 0.07$) |
| **10** | ElasticNet Regression | 0.6645 | 1149.51 | 895.34 | N/A | Balanced L1/L2 penalty |

### Classification Track (10 Models) — Target: `y` (Term Deposit Subscription)
| Rank | Algorithm | Accuracy | Precision (Weighted) | Recall (Weighted) | Weighted F1 | ROC-AUC | Category |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Gradient Boosting** | **0.8978** | **0.8803** | **0.8978** | **0.8770** | **0.8056** | Ensemble Boosting |
| **2** | **Bagging Classifier** | **0.8963** | **0.8777** | **0.8963** | **0.8736** | **0.7634** | Ensemble Bagging |
| **3** | **MLP Classifier** | **0.8945** | **0.8745** | **0.8945** | **0.8701** | **0.7828** | Neural Network |
| **4** | Support Vector Classifier (SVC) | 0.8933 | 0.8726 | 0.8933 | 0.8659 | 0.7184 | Kernel Machine |
| **5** | Random Forest | 0.8943 | 0.8758 | 0.8943 | 0.8654 | 0.8004 | Ensemble Bagging |
| **6** | Logistic Regression | 0.8933 | 0.8730 | 0.8933 | 0.8650 | 0.7717 | Linear Baseline |
| **7** | K-Nearest Neighbors (KNN) | 0.8874 | 0.8612 | 0.8874 | 0.8631 | 0.7027 | Distance-based |
| **8** | Decision Tree | 0.8927 | 0.8740 | 0.8927 | 0.8606 | 0.7055 | Tree Partitioning |
| **9** | AdaBoost | 0.8909 | 0.8739 | 0.8909 | 0.8535 | 0.7651 | Ensemble Boosting |
| **10** | Gaussian Naive Bayes | 0.8452 | 0.8615 | 0.8452 | 0.8525 | 0.7514 | Probabilistic |


### Clustering Track (Unsupervised Segmentation) — $k=4$ Clusters
| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Silhouette Score** | **0.3821** | Good cluster cohesion with distinct separation boundaries |
| **Davies-Bouldin Index** | **0.9412** | Low ratio of within-cluster scatter to between-cluster separation |
| **Calinski-Harabasz Index** | **2,684.5** | High between-cluster variance relative to within-cluster dispersion |
| **Optimal Segments** | 4 Personas | Champions / High-Spenders, Loyal Core, Promising / Recent, At-Risk / Hibernating |

---

## 6. Author Information & Academic Attribution

- **Course:** 23CSE301 — Machine Learning
- **Institution:** Amrita Vishwa Vidyapeetham
- **Academic Year:** 2026–2027
- **Deliverables:** Review 1 & Review 2 Capstone Deliverables
- **License:** MIT License
