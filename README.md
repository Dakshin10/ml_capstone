# 23CSE301 — Machine Learning Capstone | Review 1

Comprehensive machine learning capstone project implementing both **Regression** and **Classification (Part A)** tracks for academic Review 1 evaluation.

---

## 1. Project Overview & Tracks

| Track | Target Variable | Type | Dataset | Rows | Features | Primary Metric |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Regression** | `traffic_volume` | Continuous (vehicles/hr) | `dataset/Metro_Interstate_Traffic_Volume.csv` | 48,204 | 9 | $R^2$, RMSE, MAE |
| **Classification (Part A)** | `y` (`yes` / `no`) | Binary (0 / 1) | `data/bank-full.csv` | 45,211 | 15 (after exclusions) | Weighted F1, Accuracy, Confusion Matrix |

---

## 2. Dataset Descriptions

### Track 1: Metro Interstate Traffic Volume (Regression)
- **Source:** UCI Machine Learning Repository / Kaggle (`Metro_Interstate_Traffic_Volume.csv`)
- **Objective:** Predict hourly Interstate 94 Westbound traffic volume based on weather features, temperature, date/time attributes, and national holidays.
- **Raw Features:** `holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`
- **Target:** `traffic_volume` (continuous numerical target, range 0–7,280 vehicles/hour)
- **Feature Engineering:** Temporal extraction (`hour`, `day_of_week`, `month`), weekend indicator (`is_weekend`), holiday flag (`is_holiday`), and temperature conversion (`temp_celsius = temp - 273.15`).

### Track 2: Bank Marketing Term Deposit Prediction (Classification — Part A)
- **Source:** UCI Bank Marketing Dataset (`bank-full.csv`, delimited by `;`)
- **Objective:** Predict whether a client will subscribe to a term deposit (`yes` / `no`).
- **Features:** Demographics (`age`, `job`, `marital`, `education`), financial history (`default`, `balance`, `housing`, `loan`), and campaign metadata (`contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `poutcome`).
- **Target:** `y` (mapped to binary: `yes` $\rightarrow$ 1, `no` $\rightarrow$ 0; class ratio approx. 88.3% no vs. 11.7% yes).
- **Critical Feature Exclusion:** `duration` is **dropped** from the predictor set because call duration is unknown prior to call completion, constituting a realistic predictive-availability / data-leakage concern.

---

## 3. Algorithms Implemented (Review 1)

### Regression Track (`notebooks/regression.ipynb`) — First 5 Regression Algorithms
1. **Linear Regression** (Ordinary Least Squares baseline)
2. **Ridge Regression** (L2 regularized linear model, with GridSearchCV tuning)
3. **Lasso Regression** (L1 regularized linear model, with GridSearchCV tuning)
4. **ElasticNet Regression** (Combined L1/L2 regularized linear model)
5. **Polynomial Regression (degree = 2)** (Degree-2 interaction feature expansion + Linear Regression)
*Includes 5-fold Cross-Validation on top estimators, residual plots, and predicted vs actual performance charts.*

### Classification Track — Part A (`notebooks/classification.ipynb`)
1. **Logistic Regression** (Baseline linear classifier, lbfgs solver, max_iter=1000)
2. **K-Nearest Neighbors (KNN)** (Distance-based classifier, k=5 with scaled features)
3. **Gaussian Naive Bayes** (Probabilistic classifier with dense-transformation pipeline)
4. **Decision Tree Classifier** (Tree-based model with max_depth control, Gini importance analysis)
5. **Support Vector Classifier (SVC)** (RBF kernel, probability=True, standardized inputs)
*Note: Classification Part B (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP) belongs to Review 2.*

---

## 4. Preprocessing & Leakage Prevention

- **Train/Test Splitting:**
  - Regression: 80/20 train/test split with `random_state=42`.
  - Classification: Stratified 80/20 train/test split (`stratify=y`, `random_state=42`) to preserve class ratios.
- **Strict Leakage Guard:** All scalers (`StandardScaler`) and encoders (`OneHotEncoder(handle_unknown='ignore')`) are fitted **strictly on training partitions only** inside scikit-learn `Pipeline` and `ColumnTransformer` workflows.
- **Consistent Benchmarking:** The exact same held-out test split is used across all comparative models.

---

## 5. Repository Structure

```text
ml_capstone/
├── .gitignore
├── README.md
├── requirements.txt
├── dataset/
│   └── Metro_Interstate_Traffic_Volume.csv # Metro Interstate Traffic Volume dataset
├── data/
│   └── bank-full.csv                       # Bank marketing dataset
├── models/                                 # Serialized model artifacts (optional)
└── notebooks/
    ├── classification.ipynb                # Classification Track (Part A — Review 1)
    └── regression.ipynb                    # Regression Track (5 Algorithms — Review 1)
```

---

## 6. Installation & Execution

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/SH-Nihil-Mukkesh-25/ml_capstone.git
cd ml_capstone

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
Navigate to:
- `notebooks/regression.ipynb` $\rightarrow$ Click `Kernel` $\rightarrow$ `Restart & Run All`
- `notebooks/classification.ipynb` $\rightarrow$ Click `Kernel` $\rightarrow$ `Restart & Run All`

Both notebooks use relative paths (`../dataset/` and `../data/`) and run self-contained from top to bottom.

---

## 7. Academic Integrity & AI Assistance Acknowledgement

In compliance with university evaluation policies for B.Tech 23CSE301:
- Generative AI (**Google Antigravity / Gemini**) was utilized solely for code scaffolding, boilerplate generation, and notebook structuring.
- All analytical interpretations, exploratory observations, and conclusions are drafted with explicit verification tags (`DRAFT OBSERVATION — VERIFY AGAINST THE ACTUAL PLOT/REPORT`) to be confirmed against executed outputs by the project team.
- No model performance metrics, dataset statistics, or experimental results were fabricated.

---
*B.Tech CSE — 23CSE301 Machine Learning Capstone | Review 1 Submission*
