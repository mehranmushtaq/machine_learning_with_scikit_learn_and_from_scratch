# 🤖 Machine Learning with Scikit-Learn & From Scratch

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

**An end-to-end Machine Learning repository built during an intensive internship — covering supervised learning, unsupervised learning, ensemble methods, from-scratch implementations, and applied real-world projects.**

[Explore Projects](#-featured-projects) · [View Structure](#-repository-structure) · [Quick Start](#-quick-start) · [Author](#-author)

-----

## Overview

This repository is a **structured, production-minded ML learning track** developed as part of an internship program. It is not just a collection of notebooks — it is a curated journey through the fundamentals and applied practice of machine learning, built with an emphasis on **understanding before implementation**.

Each module follows a consistent pattern:

- Conceptual grounding before code
- Scikit-Learn pipeline-first design (no data leakage)
- Hyperparameter tuning via `GridSearchCV`
- Model evaluation with industry-standard metrics
- Clean, commented notebooks suitable for portfolio review

-----

## Repository Structure

```
machine_learning_with_scikit_learn_and_from_scratch/
│
├── 📁 Datasets/                              # Shared datasets used across modules
│   ├── Emotion_classify_Data.csv
│   ├── Iris.csv
│   ├── Social_Network_Ads.csv
│   ├── house_prices_practice.csv
│   └── insurance.csv
│
├── 📁 linear_regression/
│   ├── Linear_regression.ipynb
│   └── README.md
│
├── 📁 Logistic Regression/
│   ├── Logistic_Regressor.ipynb
│   └── README.md
│
├── 📁 KNN/
│   ├── Knn.ipynb
│   └── README.md
│
├── 📁 Decision Tree/
│   ├── decision_tree_classifier.ipynb
│   ├── decision_tree_regressor.ipynb
│   └── README.md
│
├── 📁 Naive Bayes/
│   ├── naive_bayes.ipynb
│   └── README.md
│
├── 📁 Support Vector Machine/
│   ├── svc.ipynb
│   ├── svr.ipynb
│   └── README.md
│
├── 📁 Regularization (Lasso/Ridge)/
│   ├── lasso_ridge.ipynb
│   └── README.md
│
├── 📁 ensemble learning/
│   ├── bagging/
│   │   ├── Random_forest.ipynb
│   │   └── README.md
│   └── boosting/
│       ├── gradient_boosting.ipynb
│       ├── ada_boosting.ipynb
│       ├── xgboost.ipynb
│       └── README.md
│
├── 📁 ml-from-scratch/                       # Algorithms without any sklearn
│   ├── linear_reg.ipynb
│   ├── logistic_reg.ipynb
│   ├── knn_regressor.ipynb
│   └── README.md
│
├── 📁 unsupervised ml/
│   ├── k_means.ipynb
│   ├── k_means_clustering.ipynb
│   ├── hiearchichal_clustering.ipynb
│   ├── dbscan.ipynb
│   ├── README.md
│   └── 📁 projects/
│       └── 📁 thyroid_outlier_detection/    
│           ├── thyroid_outlier_detection.ipynb
│           ├── thyroid_dataset.csv
│           └── README.md
│
├── 📁 Projects/                              
│   ├── 📁 CreditWise Loan System/
│   │   ├── Loan_Approval.ipynb
│   │   ├── loan_approval_data.csv
│   │   └── README.md
│   ├── 📁 ecommerce-purchase-prediction/
│   │   ├── predicting_ecommerce.ipynb
│   │   ├── shop_smart_ecommerce.csv
│   │   └── README.md
│   └── 📁 disease_prediction_pipeline/
│       ├── disease_prediction_pipeline.ipynb
│       ├── novagen_dataset.csv
│       └── README.md
│
├── notebook_vs_production.md
├── requirements.txt
└── README.md                                 
```

-----

## Curriculum Coverage

### Supervised Learning

|Module                         |Algorithm                           |Key Concepts                                       |
|-------------------------------|------------------------------------|---------------------------------------------------|
|`linear_regression/`           |Ordinary Least Squares, Ridge, Lasso|Coefficients, MSE, R², Regularization              |
|`Logistic Regression/`         |Binary & Multi-class LR             |Sigmoid, Cross-Entropy, Polynomial Features        |
|`KNN/`                         |K-Nearest Neighbours                |Distance metrics, K tuning, Curse of Dimensionality|
|`Decision Tree/`               |CART Classifier & Regressor         |Gini, Entropy, Pre/Post Pruning                    |
|`Naive Bayes/`                 |Gaussian Naive Bayes                |Bayes Theorem, Conditional Independence            |
|`Support Vector Machine/`      |SVC + SVR                           |Kernels (RBF, Poly), Margin, C & γ tuning          |
|`Regularization (Lasso/Ridge)/`|L1 / L2 Penalty                     |Bias-Variance Trade-off                            |

### Ensemble Methods

|Module                       |Technique                           |Highlights                        |
|-----------------------------|------------------------------------|----------------------------------|
|`ensemble learning/bagging/` |Random Forest                       |Feature Importance, OOB Score     |
|`ensemble learning/boosting/`|AdaBoost, Gradient Boosting, XGBoost|Sequential learners, Learning rate|

### Unsupervised Learning

|Module                                               |Algorithm                    |Key Concepts                                         |
|-----------------------------------------------------|-----------------------------|-----------------------------------------------------|
|`unsupervised ml/`                                   |K-Means, Hierarchical, DBSCAN|Elbow Method, Dendrograms, Epsilon, Outlier Detection|
|`unsupervised ml/projects/thyroid_outlier_detection/`|Isolation Forest + LOF       |Anomaly scoring, Medical outlier detection           |

### ML From Scratch (Pure Python / NumPy)

|Module                               |What’s Implemented                             |
|-------------------------------------|-----------------------------------------------|
|`ml-from-scratch/linear_reg.ipynb`   |Gradient Descent, cost function, weight updates|
|`ml-from-scratch/logistic_reg.ipynb` |Sigmoid, binary cross-entropy, manual backprop |
|`ml-from-scratch/knn_regressor.ipynb`|Euclidean distance, k-neighbor voting          |

-----

##  Featured Projects

### Disease Prediction Pipeline

> `Projects/disease_prediction_pipeline/` · Voting Classifier Ensemble · 9,800 patients

A production-ready clinical classification system built for **NovaGen Research Labs**. Uses a hard-voting ensemble of Logistic Regression, Random Forest, and Naïve Bayes inside a full `sklearn.Pipeline` with stratified cross-validation.

- **Accuracy:** 94.89%
- **CV Recall:** 95.46%
- **Use Case:** Clinical trial participant screening

-----

### E-Commerce Purchase Prediction

> `Projects/ecommerce-purchase-prediction/` · Decision Tree · 12,330 sessions

Predicts visitor-to-buyer conversion from behavioral session data. Addresses severe class imbalance (85/15 split) with `class_weight='balanced'` and depth pruning.

- **F1-Score:** 0.6485 *(benchmark: 0.55 — +18% improvement)*
- **Use Case:** Marketing conversion optimisation

-----

### CreditWise Loan Approval System

> `Projects/CreditWise Loan System/` · Classification Pipeline · End-to-end

Automated loan risk classification using financial and demographic features. Full preprocessing pipeline with feature engineering, encoding, scaling, and model evaluation.

-----

### Thyroid Outlier Detection 

> `unsupervised ml/projects/thyroid_outlier_detection/` · Isolation Forest + LOF · 1,000 patients

Detects anomalous thyroid hormone profiles in patient lab data using unsupervised anomaly detection. Combines Isolation Forest and Local Outlier Factor to flag clinically significant outliers without labeled training data.

- **Precision@K:** 89%
- **Use Case:** Automated lab result triage, rare disease flagging

-----

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mehranmushtaq/machine_learning_with_scikit_learn_and_from_scratch.git

# 2. Navigate into the project
cd machine_learning_with_scikit_learn_and_from_scratch

# 3. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch Jupyter
jupyter notebook
```

-----

## Tech Stack

|Tool                |Version|Purpose                  |
|--------------------|-------|-------------------------|
|Python              |3.10+  |Core language            |
|Scikit-Learn        |1.x    |ML algorithms & pipelines|
|XGBoost             |1.7+   |Gradient boosting        |
|Pandas              |1.5+   |Data manipulation        |
|NumPy               |1.23+  |Numerical computing      |
|Matplotlib / Seaborn|Latest |Visualisation            |
|Jupyter Notebook    |Latest |Interactive development  |

-----

## Core Engineering Practices

```
✅ sklearn.Pipeline used throughout — prevents data leakage at every stage
✅ GridSearchCV for principled hyperparameter optimisation
✅ Stratified K-Fold Cross-Validation for robust generalisation estimates
✅ ColumnTransformer for heterogeneous feature preprocessing
✅ Class imbalance handling (class_weight, resampling strategies)
✅ Evaluation beyond accuracy — F1, Precision, Recall, AUC-ROC, Confusion Matrix
✅ Feature importance analysis and interpretability
✅ From-scratch implementations to verify theoretical understanding
✅ Clean, reproducible notebooks with random_state seeding
```

-----

⭐ **If this repository helped you, consider giving it a star!**


*“First, solve the problem. Then, write the code.”* — John Johnson
