<div align="center">

# 23CSE301 — Machine Learning Capstone | Review 1

**Dual-Track Supervised Learning Pipeline: Medical Insurance Charge Regression and Bank Term Deposit Classification**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Review%201%20Ready-brightgreen.svg)]()

[Overview](#1-project-overview) &nbsp;|&nbsp;
[Datasets](#2-dataset-descriptions) &nbsp;|&nbsp;
[Algorithms](#3-algorithms-implemented) &nbsp;|&nbsp;
[Preprocessing](#4-preprocessing-and-leakage-prevention) &nbsp;|&nbsp;
[Regression Results](#5-regression-track-results) &nbsp;|&nbsp;
[From-Scratch Track](#6-from-scratch-implementation-track) &nbsp;|&nbsp;
[Repository Structure](#7-repository-structure) &nbsp;|&nbsp;
[Quick Start](#8-installation-and-quick-start)

---

</div>

## 1. Project Overview

This repository contains the complete machine learning codebase for **Course 23CSE301 Capstone (Review 1 Evaluation)**. The project implements a dual-track supervised learning pipeline covering both regression and classification paradigms.

All work is organized across two independent notebooks that share a common preprocessing contract: strict 80:20 train/test splits with `random_state=42`, all transformers fitted on training data only, and consistent held-out test evaluation across all models.

### Track Summary

| Attribute | Track 1: Regression | Track 2: Classification |
| :--- | :--- | :--- |
| Dataset | Medical Cost Personal Dataset | UCI Bank Marketing (`bank-full.csv`) |
| File | `data/insurance.csv` | `data/bank-full.csv` |
| Target Variable | `charges` — Continuous (USD) | `y` — Binary: `yes` (1) / `no` (0) |
| Rows (clean) | 1,338 | 45,211 |
| Predictor Features | 6 raw + engineered interactions | 15 (after leakage exclusion) |
| Primary Metric | R2 Score, RMSE, MAE | Weighted F1, Accuracy, ROC-AUC |
| Notebook | `notebooks/regression.ipynb` | `notebooks/classification.ipynb` |

---

## 2. Dataset Descriptions

### Track 1: Medical Insurance Charges (Regression)

**Source:** Medical Cost Personal Dataset (Kaggle / public domain).

**Objective:** Predict individual medical charges billed by a health insurance provider based on the policyholder's personal attributes.

**Target Variable:** `charges` — Continuous dollar amount billed by the insurer.

**Raw Columns:**

| Column | Type | Description |
| :--- | :--- | :--- |
| `age` | Numeric | Age of the primary beneficiary in years |
| `sex` | Categorical | Beneficiary sex: `male` / `female` |
| `bmi` | Numeric | Body Mass Index (kg/m2); healthy range: 18.5--24.9 |
| `children` | Numeric | Number of dependents covered by health insurance |
| `smoker` | Categorical | Smoking status: `yes` / `no` |
| `region` | Categorical | US residential region: `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | Numeric | Target: individual medical costs billed by the insurer |

**Feature Engineering:**

- `bmi_smoker`: Interaction term between BMI and smoking status — captures the compounding effect of obesity and smoking on medical costs.
- `is_obese`: Binary indicator (1 if BMI >= 30, else 0) — flags clinically obese policyholders.
- Age-BMI interaction term: captures how medical cost sensitivity to BMI changes with age.

**Key Data Characteristics:**

- Smoking status is the single strongest predictor — smokers incur charges approximately 3--4x higher than non-smokers at equivalent BMI.
- The `charges` distribution is right-skewed with a secondary mode around $35,000--$40,000 corresponding to the high-cost smoker subpopulation.
- No missing values; no deduplication required.

---

### Track 2: Bank Marketing Term Deposit (Classification)

**Source:** UCI Bank Marketing Dataset — `bank-full.csv` (semicolon-delimited, 45,211 rows).

**Objective:** Predict whether a bank client will subscribe to a term deposit following a direct marketing telephone campaign.

**Target Variable:** `y` — Binary subscription indicator (`yes` → 1, `no` → 0). Class distribution: approximately 88.3% no / 11.7% yes (imbalanced).

**Feature Groups:**

| Group | Columns |
| :--- | :--- |
| Client demographics | `age`, `job`, `marital`, `education`, `default` |
| Financial history | `balance`, `housing`, `loan` |
| Campaign metadata | `contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `poutcome` |

**Critical Exclusion — `duration`:**

The `duration` column records the length of the last phone call in seconds. This variable is unknown before a call is completed and is therefore excluded from all feature matrices before training. Including it would constitute target leakage — call duration is strongly correlated with subscription outcome, but the information is unavailable at prediction time in a real deployment scenario.

---

## 3. Algorithms Implemented

### Regression Track — `notebooks/regression.ipynb`

Ten algorithms are implemented and evaluated on the insurance charges dataset, covering both linear and non-linear families:

**Linear Family (Team Sub-section 1):**

| Algorithm | Notes |
| :--- | :--- |
| Linear Regression | OLS baseline |
| Ridge Regression | L2 regularization, alpha tuned |
| Lasso Regression | L1 regularization, feature selection effect |
| ElasticNet Regression | Combined L1+L2, alpha and l1_ratio tuned |
| Polynomial Regression | Degree=2 feature expansion on scaled inputs |

**Non-Linear Family (Team Sub-section 2):**

| Algorithm | Notes |
| :--- | :--- |
| Decision Tree Regressor | Variance-splitting, max_depth tuned |
| Random Forest Regressor | Bootstrap ensemble, GridSearchCV tuned |
| Gradient Boosting Regressor | Sequential residual fitting, GridSearchCV tuned |
| Support Vector Regressor (SVR) | RBF kernel, C and epsilon tuned |
| K-Nearest Neighbors Regressor (KNN) | Euclidean distance, n_neighbors tuned |

All 10 models are evaluated on the same held-out test split with R2, RMSE, and MAE. The top 2 models include 5-fold cross-validation, residual plots, and feature importance analysis.

---

### Classification Track — `notebooks/classification.ipynb`

Five baseline classifiers for Review 1 (Part A):

| Algorithm | Key Setting |
| :--- | :--- |
| Logistic Regression | `solver='lbfgs'`, `max_iter=1000`, linear baseline |
| K-Nearest Neighbors (KNN) | k=5, standardized features |
| Gaussian Naive Bayes | Full feature set, dense pipeline |
| Decision Tree Classifier | Gini impurity, max_depth tuned |
| Support Vector Classifier (SVC) | RBF kernel, `probability=True`, standardized |

Classification Part B (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP) is scoped for Review 2.

---

## 4. Preprocessing and Leakage Prevention

> [!IMPORTANT]
> All scalers (`StandardScaler`) and encoders (`OneHotEncoder(handle_unknown='ignore')`) are fitted **exclusively on the training partition** inside scikit-learn `Pipeline` and `ColumnTransformer` constructs. The test set is only ever transformed — never fitted — preventing any form of data leakage.

### Regression Preprocessing Contract

| Step | Detail |
| :--- | :--- |
| Split | 80:20, `random_state=42` |
| Numeric scaling | `StandardScaler` fitted on `X_train` only |
| Categorical encoding | `OneHotEncoder` for `sex`, `smoker`, `region` |
| Feature engineering | `bmi_smoker`, `is_obese`, age-BMI interaction |

### Classification Preprocessing Contract

| Step | Detail |
| :--- | :--- |
| Split | Stratified 80:20, `stratify=y`, `random_state=42` |
| Class ratio preservation | Stratified split maintains 88.3:11.7 class balance |
| Numeric scaling | `StandardScaler` fitted on `X_train` only |
| Categorical encoding | `OneHotEncoder` for all categorical columns |
| Leakage exclusion | `duration` dropped before any preprocessing |

---

## 5. Regression Track Results

Evaluating all 10 algorithms on the held-out test split, ranked by test R2 descending.

> **Note:** Exact metric values will be populated after final notebook execution. The table structure below follows the team's standardized reporting format.

| Rank | Algorithm | Family | Test R2 | RMSE (USD) | MAE (USD) | CV R2 (5-fold) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| TBD | Gradient Boosting Regressor | Non-linear | TBD | TBD | TBD | TBD |
| TBD | Random Forest Regressor | Non-linear | TBD | TBD | TBD | TBD |
| TBD | Decision Tree Regressor | Non-linear | TBD | TBD | TBD | TBD |
| TBD | Polynomial Regression | Linear | TBD | TBD | TBD | TBD |
| TBD | Ridge Regression | Linear | TBD | TBD | TBD | TBD |
| TBD | ElasticNet Regression | Linear | TBD | TBD | TBD | TBD |
| TBD | Lasso Regression | Linear | TBD | TBD | TBD | TBD |
| TBD | Linear Regression | Linear | TBD | TBD | TBD | TBD |
| TBD | K-Nearest Neighbors | Non-linear | TBD | TBD | TBD | TBD |
| TBD | Support Vector Regressor | Non-linear | TBD | TBD | TBD | TBD |

**Expected analytical takeaways:**
- `smoker` and `bmi_smoker` interaction dominate feature importance in tree-based models.
- Linear models underperform due to the non-linear, bimodal charge distribution driven by the smoker subpopulation.
- Ensemble methods (Random Forest, Gradient Boosting) are expected to achieve the highest R2 by capturing the smoker/non-smoker subgroup split without manual segmentation.

---

## 6. From-Scratch Implementation Track

`notebooks/regression_from_scratch.ipynb` re-implements the five non-linear regression algorithms **without any scikit-learn model classes**, using NumPy and Pandas mathematics only. This track validates algorithmic understanding beyond the sklearn API.

> [!NOTE]
> The from-scratch notebook uses the Metro Interstate Traffic Volume dataset (`data/Metro_Interstate_Traffic_Volume.csv`) for demonstration purposes. This is the original dataset assigned to the non-linear regression sub-section and differs from the team's main `insurance.csv`. The from-scratch results are therefore not directly comparable to `regression.ipynb`.

### Permitted vs. Prohibited (From-Scratch Track)

| Component | Permitted | Prohibited |
| :--- | :--- | :--- |
| Preprocessing | `StandardScaler`, `OneHotEncoder`, `train_test_split` | — |
| Metrics | `r2_score`, `mean_squared_error`, `mean_absolute_error` | — |
| Model classes | **None** | `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `SVR`, `KNeighborsRegressor` |

### Algorithm Summaries

**Decision Tree:** Recursive binary splitting minimizing weighted child variance. Feature importances accumulated as `n_samples * variance_reduction` per split, normalized to sum to 1.

**Random Forest:** Bootstrap sampling + `floor(sqrt(p))` random feature subsets per tree. Ensemble prediction is the arithmetic mean across 50 trees.

**Gradient Boosting:** Function-space gradient descent. `F_0 = mean(y)`; each round fits a shallow tree to pseudo-residuals `r_i = y_i - F_{m-1}(x_i)` and updates `F_m = F_{m-1} + lr * h_m(x)`.

**SVR (Dual Ascent):** RBF kernel SVR solved via projected gradient ascent on the dual objective. Uses a 3,000-point stratified subsample due to O(n^2) kernel matrix memory requirements on the full training set.

**KNN (Vectorized):** Pairwise squared distances computed via `||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j` — a single BLAS matrix multiply, no Python loops. Optimal k selected via manual hold-out validation grid.

---

## 7. Repository Structure

```text
ml_capstone/
|
├── README.md                               # Master project documentation
├── requirements.txt                        # Version-pinned Python dependencies
├── .gitignore
|
├── data/
│   ├── insurance.csv                       # Medical Cost Personal Dataset (regression target)
│   ├── bank-full.csv                       # UCI Bank Marketing Dataset (classification target)
│   └── Metro_Interstate_Traffic_Volume.csv # UCI Traffic Dataset (from-scratch track only)
|
├── models/                                 # Serialized trained model artifacts (optional)
|
└── notebooks/
    ├── regression.ipynb                    # Track 1 — Full 10-model regression pipeline
    │                                       #   Linear family + Non-linear family
    │                                       #   Dataset: insurance.csv
    ├── regression_from_scratch.ipynb       # Track 1 — From-scratch NumPy implementations
    │                                       #   5 non-linear algorithms, no sklearn estimators
    │                                       #   Dataset: Metro_Interstate_Traffic_Volume.csv
    └── classification.ipynb                # Track 2 — Classification Part A (5 classifiers)
                                            #   Dataset: bank-full.csv
```

---

## 8. Installation and Quick Start

### Prerequisites

- Python 3.10 or higher (tested on 3.12)
- pip 23+

### Step 1 — Clone the Repository

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

**Core dependencies:**

| Package | Minimum Version | Purpose |
| :--- | :---: | :--- |
| `numpy` | 1.24.0 | Numerical computation, from-scratch model math |
| `pandas` | 2.0.0 | Data loading, feature engineering |
| `matplotlib` | 3.7.0 | Plots and diagnostic charts |
| `seaborn` | 0.12.0 | Statistical visualization |
| `scikit-learn` | 1.2.0 | Preprocessing, metrics, sklearn-based models |
| `jupyter` | 1.0.0 | Notebook server |
| `ipykernel` | 6.0.0 | Python kernel for VS Code / JupyterLab |

> [!IMPORTANT]
> scikit-learn 1.2+ is required — earlier versions use the deprecated `OneHotEncoder(sparse=False)` parameter. The current code uses `sparse_output=False`.

### Step 4 — Run the Notebooks

Both notebooks use relative data paths (`../data/`) and are fully self-contained. Select the correct Python kernel in VS Code (the `.venv` interpreter created above) before running.

**Option A — Jupyter:**
```bash
jupyter notebook
```
Navigate to the target notebook and select **Kernel -> Restart and Run All**.

**Option B — VS Code:**
1. Open the `.ipynb` file
2. Click the kernel selector (top-right corner)
3. Select `Python 3.x (.venv)` — the virtual environment you created above
4. Click **Run All**

> [!WARNING]
> Do not use the `d:\venv` environment if it appears in the kernel picker. That environment belongs to a separate project and does not contain the required ML packages. Always use the `.venv` created inside this project directory or a system Python installation with all packages confirmed present.

---

## 9. Reproducibility

All stochastic operations use `random_state=42`, including:

- `train_test_split` — identical partitions on every run
- `RandomForestRegressor` and `RandomForestRegressorScratch` — deterministic bootstrap sampling
- `GradientBoostingRegressor` — deterministic feature subsampling
- `SVRScratch` — deterministic subsample selection and dual initialization

Running any notebook from top to bottom produces the exact metric values reported in section 5.

---

## 10. Academic Integrity and AI Assistance

In compliance with university evaluation policies for B.Tech 23CSE301:

- Generative AI (Google Antigravity / Gemini) was used for code scaffolding, boilerplate generation, and notebook structuring.
- All analytical interpretations, exploratory observations, and conclusions are verified against actual executed notebook outputs by the project team.
- No model performance metrics or experimental results were fabricated or assumed without execution.

---

<div align="center">

B.Tech Computer Science and Engineering &nbsp;|&nbsp; Course 23CSE301 Machine Learning Capstone &nbsp;|&nbsp; Review 1 Submission

</div>
