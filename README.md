# 🤖 Machine Learning with Scikit-Learn & From Scratch

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](https://github.com/mehranmushtaq/ml-scikit-scratch)

**An end-to-end Machine Learning repository covering supervised learning, unsupervised learning, ensemble methods, from-scratch implementations, and applied real-world projects — built as part of an intensive internship program.**

[Featured Projects](#-featured-projects) · [Curriculum](#-curriculum-coverage) · [Repo Structure](#-repository-structure) · [Quick Start](#-quick-start) · [Author](#-author)


##  Overview

This repository is a **structured, production-minded ML learning track** developed during an internship. It is not just a collection of notebooks — it is a curated journey through the fundamentals and applied practice of machine learning, built with emphasis on **understanding before implementation**.

Each module follows a consistent pattern:

- Conceptual grounding before code
- Scikit-Learn pipeline-first design (no data leakage)
- Hyperparameter tuning via `GridSearchCV`
- Model evaluation with industry-standard metrics
- Clean, commented notebooks suitable for portfolio review

-----

##  Repository Structure

```
ml-scikit-scratch/
│
├── 📁 Datasets/                              # Shared datasets across modules
│   ├── Emotion_classify_Data.csv
│   ├── Iris.csv
│   ├── Social_Network_Ads.csv
│   ├── house_prices_practice.csv
│   └── insurance.csv
│
├── 📁 linear_regression/
├── 📁 Logistic Regression/
├── 📁 KNN/
├── 📁 Decision tree/
├── 📁 Naive bayes/
├── 📁 Support Vector Machine/
├── 📁 Regularizaton(Lasso:Ridge)/
├── 📁 ensemble learning/
│   ├── bagging/                              # Random Forest
│   └── boosting/                             # AdaBoost, Gradient Boosting, XGBoost
│
├── 📁 ml-from-scratch/                       # Pure Python/NumPy implementations
│   ├── linear_reg.ipynb
│   ├── logistic_reg.ipynb
│   └── knn_regressor.ipynb
│
├── 📁 unsupervised ml/
│   ├── k_means.ipynb
│   ├── hiearchichal_clustering.ipynb
│   └── dbscan.ipynb
│
├── 📁 Projects/                              # End-to-end real-world applications
│   ├── 📁 customer-segmentation-smartcart/  # NEW — Unsupervised segmentation
│   ├── 📁 CreditWise Loan System/
│   ├── 📁 ecommerce-purchase-prediction/
│   ├── 📁 thyroid_outlier_detection/
│   └── 📁 disease_prediction_pipeline/
│
├── notebook_vs_production.md
├── requirements.txt
└── README.md
```

-----

##  Curriculum Coverage

### Supervised Learning

|Module                       |Algorithm                  |Key Concepts                                       |
|-----------------------------|---------------------------|---------------------------------------------------|
|`linear_regression/`         |OLS, Ridge, Lasso          |Coefficients, MSE, R², Regularization              |
|`Logistic Regression/`       |Binary & Multi-class LR    |Sigmoid, Cross-Entropy, Polynomial Features        |
|`KNN/`                       |K-Nearest Neighbours       |Distance metrics, K-tuning, Curse of Dimensionality|
|`Decision Tree/`             |CART Classifier & Regressor|Gini, Entropy, Pre/Post Pruning                    |
|`Naive Bayes/`               |Gaussian Naive Bayes       |Bayes Theorem, Conditional Independence            |
|`Support Vector Machine/`    |SVC + SVR                  |Kernels (RBF, Poly), Margin, C & γ tuning          |
|`Regularizaton(Lasso:Ridge)/`|L1 / L2 Penalty            |Bias-Variance Trade-off                            |

### Ensemble Methods

|Module                       |Technique                           |Highlights                        |
|-----------------------------|------------------------------------|----------------------------------|
|`ensemble learning/bagging/` |Random Forest                       |Feature Importance, OOB Score     |
|`ensemble learning/boosting/`|AdaBoost, Gradient Boosting, XGBoost|Sequential learners, Learning Rate|

### Unsupervised Learning

|Module            |Algorithm                    |Key Concepts                      |
|------------------|-----------------------------|----------------------------------|
|`unsupervised ml/`|K-Means, Hierarchical, DBSCAN|Elbow Method, Dendrograms, Epsilon|

### ML From Scratch (Pure Python / NumPy)

|Module               |What’s Implemented                             |
|---------------------|-----------------------------------------------|
|`linear_reg.ipynb`   |Gradient Descent, cost function, weight updates|
|`logistic_reg.ipynb` |Sigmoid, binary cross-entropy, manual backprop |
|`knn_regressor.ipynb`|Euclidean distance, k-neighbor voting          |

-----

## Featured Projects

###  SmartCart Customer Segmentation  `NEW`

> `Projects/customer-segmentation-smartcart/`  ·  Unsupervised Learning  ·  K-Means + Agglomerative Clustering + PCA

End-to-end customer segmentation pipeline for a retail platform. Discovers **4 distinct customer personas** from behavioral and demographic data — enabling personalized marketing and product strategy without any labeled data.

**Full Pipeline:**

```
Raw Data → Missing Value Imputation → Feature Engineering → Outlier Removal
→ One-Hot Encoding → StandardScaler → PCA (4 components, ~55% variance)
→ Elbow + Silhouette K-Selection → K-Means & Agglomerative Clustering → Cluster Analysis
```

**Cluster Profiles (Agglomerative — Final Model):**

|Cluster|Avg Income|Avg Spend|Profile                           |Recommended Strategy           |
|-------|----------|---------|----------------------------------|-------------------------------|
|0      |~$42,706  |~$327    |Budget-conscious, older families  |Bundle deals, discount promos  |
|1      |~$66,279  |~$1,055  |Affluent loyalists, catalog buyers|Premium rewards, upsells       |
|2      |~$35,326  |~$110    |Low-income, high web visits       |Re-engagement, web-first offers|
|3      |~$74,727  |~$1,271  |Premium high-spenders             |VIP programs, early access     |

**Dataset:** 2,236 customers × 15 features  |  **Models:** K-Means & Agglomerative (Ward linkage), k=4

-----

### Disease Prediction Pipeline

> `Projects/disease_prediction_pipeline/`  ·  Voting Classifier Ensemble  ·  9,800 patients

Production-ready clinical classification system built for **NovaGen Research Labs**. Hard-voting ensemble of Logistic Regression, Random Forest, and Naïve Bayes inside a full `sklearn.Pipeline` with stratified cross-validation.

- **Accuracy:** 94.89%  |  **CV Recall:** 95.46%
- **Use Case:** Clinical trial participant screening

-----

### E-Commerce Purchase Prediction

> `Projects/ecommerce-purchase-prediction/`  ·  Decision Tree  ·  12,330 sessions

Predicts visitor-to-buyer conversion from behavioral session data. Handles severe class imbalance (85/15 split) with `class_weight='balanced'` and depth pruning.

- **F1-Score:** 0.6485 *(+18% over 0.55 baseline)*
- **Use Case:** Marketing conversion optimisation

-----

### CreditWise Loan Approval System

> `Projects/CreditWise Loan System/`  ·  Classification Pipeline

Automated loan risk classification using financial and demographic features. Full preprocessing pipeline with feature engineering, encoding, scaling, and model evaluation. Includes a production `.py` script alongside the notebook.

-----

###  Thyroid Outlier Detection

> `Projects/thyroid_outlier_detection/`  ·  Isolation Forest + LOF  ·  1,000 patients

Detects anomalous thyroid hormone profiles in patient lab data using unsupervised anomaly detection — no labeled training data required.

- **Precision@K:** 89%
- **Use Case:** Automated lab result triage, rare disease flagging

-----

##  Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mehranmushtaq/ml-scikit-scratch.git

# 2. Navigate into the project
cd ml-scikit-scratch

# 3. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch Jupyter
jupyter notebook
```

-----

## 🧰 Tech Stack

|Tool                |Version|Purpose                  |
|--------------------|-------|-------------------------|
|Python              |3.10+  |Core language            |
|Scikit-Learn        |1.x    |ML algorithms & pipelines|
|XGBoost             |1.7+   |Gradient boosting        |
|Pandas              |1.5+   |Data manipulation        |
|NumPy               |1.23+  |Numerical computing      |
|Matplotlib / Seaborn|Latest |Visualisation            |
|Kneed               |Latest |Automated elbow detection|
|Jupyter Notebook    |Latest |Interactive development  |

-----

## 🏗️ Core Engineering Practices

```
✅ sklearn.Pipeline throughout — zero data leakage at any stage
✅ GridSearchCV for principled hyperparameter optimisation
✅ Stratified K-Fold Cross-Validation for robust generalisation estimates
✅ ColumnTransformer for heterogeneous feature preprocessing
✅ Class imbalance handling (class_weight, resampling strategies)
✅ Evaluation beyond accuracy — F1, Precision, Recall, AUC-ROC, Confusion Matrix
✅ Unsupervised validation — Elbow Method + Silhouette Score
✅ Dimensionality reduction via PCA before clustering
✅ From-scratch implementations to verify theoretical understanding
✅ Clean, reproducible notebooks with random_state seeding throughout
```

-----

##  Author

**Mehran Mushtaq**

[![GitHub](https://img.shields.io/badge/GitHub-mehranmushtaq-181717?style=flat-square&logo=github)](https://github.com/mehranmushtaq)

-----


⭐ **If this repository helped you, consider giving it a star!**

*“First, solve the problem. Then, write the code.” — John Johnson*




