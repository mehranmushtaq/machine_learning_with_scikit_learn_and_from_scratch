# 🧠 ML Scikit-Scratch

### End-to-end Machine Learning — from mathematical foundations to production pipelines

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()



> **94.89% accuracy** on clinical disease prediction  ·  **+18% F1** above e-commerce baseline  ·  **0.9975 AUC-ROC** on ensemble classification  ·  **89% Precision@K** on unsupervised anomaly detection


[Algorithms](#-algorithms)  ·  [Projects](#-projects)  ·  [From Scratch](#-ml-from-scratch)  ·  [Quick Start](#-quick-start)  ·  [Structure](#-repository-structure)

-----

## What Is This?

This repository is a **structured, end-to-end ML learning track** built during an intensive internship. It covers the full arc of machine learning — from handwriting algorithms in pure NumPy, to building production-grade scikit-learn pipelines, to solving real-world classification and anomaly detection problems.

Every module follows the same discipline:

- Understand the math before importing the library
- Build with `sklearn.Pipeline` to prevent data leakage at every step
- Tune with `GridSearchCV` — no manual guesswork
- Evaluate beyond accuracy: F1, AUC-ROC, Precision@K, confusion matrices
- Apply to real datasets with real constraints: class imbalance, unlabeled data, clinical stakes

-----

## 📂 Repository Structure

```
ml-scikit-scratch/
│
├── 📁 datasets/                          # Shared datasets used across modules
│   ├── Emotion_classify_Data.csv
│   ├── Iris.csv
│   ├── Social_Network_Ads.csv
│   ├── house_prices_practice.csv
│   └── insurance.csv
│
├── 📁 supervised/                        # Core supervised learning algorithms
│   ├── 📁 linear_regression/
│   │   ├── linear_regression.ipynb
│   │   └── README.md
│   ├── 📁 logistic_regression/
│   │   ├── logistic_regression.ipynb
│   │   └── README.md
│   ├── 📁 knn/
│   │   ├── knn.ipynb
│   │   └── README.md
│   ├── 📁 decision_tree/
│   │   ├── decision_tree_classifier.ipynb
│   │   ├── decision_tree_regressor.ipynb
│   │   └── README.md
│   ├── 📁 naive_bayes/
│   │   ├── naive_bayes.ipynb
│   │   └── README.md
│   ├── 📁 support_vector_machine/
│   │   ├── svc.ipynb
│   │   ├── svr.ipynb
│   │   └── README.md
│   └── 📁 regularization/
│       ├── lasso_ridge.ipynb
│       └── README.md
│
├── 📁 ensemble_learning/                 # Ensemble methods
│   ├── 📁 bagging/
│   │   ├── random_forest.ipynb
│   │   └── README.md
│   ├── 📁 boosting/
│   │   ├── adaboost.ipynb
│   │   ├── gradient_boosting.ipynb
│   │   ├── xgboost.ipynb
│   │   └── README.md
│   └── 📁 heterogeneous_ensemble/
│       ├── heterogeneous_ensemble_methods.ipynb
│       └── README.md
│
├── 📁 unsupervised/                      # Unsupervised learning
│   ├── k_means.ipynb
│   ├── k_means_clustering.ipynb
│   ├── hierarchical_clustering.ipynb
│   ├── dbscan.ipynb
│   └── README.md
│
├── 📁 ml_from_scratch/                   # Algorithms in pure Python / NumPy only
│   ├── linear_regression.ipynb
│   ├── logistic_regression.ipynb
│   ├── knn_regressor.ipynb
│   └── README.md
│
├── 📁 projects/                          # End-to-end applied projects
│   ├── 📁 disease_prediction_pipeline/
│   │   ├── disease_prediction_pipeline.ipynb
│   │   ├── novagen_dataset.csv
│   │   └── README.md
│   ├── 📁 ecommerce_purchase_prediction/
│   │   ├── predicting_ecommerce.ipynb
│   │   ├── predicting_ecommerce.py
│   │   ├── shop_smart_ecommerce.csv
│   │   └── README.md
│   ├── 📁 creditwise_loan_approval/
│   │   ├── loan_approval.ipynb
│   │   ├── loan_approval.py
│   │   ├── loan_approval_data.csv
│   │   └── README.md
│   ├── 📁 customer-segmentation-smartcart/
│   │   ├── customer-segmentation-smartcart.ipynb
│   │   ├── smartcart_customers.csv
│   │   └── README.md
│   └── 📁 thyroid_outlier_detection/
│       ├── thyroid_outlier_detection.ipynb
│       ├── thyroid_dataset.csv
│       └── README.md
│
├── notebook_vs_production.md             # Guide: converting notebooks to scripts
├── requirements.txt
└── README.md
```

> **Convention:** all folder and notebook names use `lowercase_with_underscores` consistently across every module.

-----

## 📐 Algorithms

### Supervised Learning

|Module                              |Algorithm                  |Key Topics                                                |
|------------------------------------|---------------------------|----------------------------------------------------------|
|`supervised/linear_regression/`     |Ordinary Least Squares     |Coefficients, MSE, R², multicollinearity                  |
|`supervised/logistic_regression/`   |Binary & Multi-class LR    |Sigmoid, log-loss, polynomial features                    |
|`supervised/knn/`                   |K-Nearest Neighbours       |Distance metrics, K selection, curse of dimensionality    |
|`supervised/decision_tree/`         |CART Classifier & Regressor|Gini, Entropy, pre/post pruning, max_depth                |
|`supervised/naive_bayes/`           |Gaussian Naive Bayes       |Bayes theorem, conditional independence, Laplace smoothing|
|`supervised/support_vector_machine/`|SVC + SVR                  |RBF/Poly kernels, C & γ tuning, margin maximisation       |
|`supervised/regularization/`        |Lasso (L1) & Ridge (L2)    |Bias-variance tradeoff, coefficient shrinkage             |

### Ensemble Methods

|Module                                     |Technique               |Highlights                                                  |
|-------------------------------------------|------------------------|------------------------------------------------------------|
|`ensemble_learning/bagging/`               |Random Forest           |Feature importance, OOB score, bootstrapping                |
|`ensemble_learning/boosting/`              |AdaBoost · GBM · XGBoost|Sequential weak learners, learning rate, early stopping     |
|`ensemble_learning/heterogeneous_ensemble/`|Voting · Stacking       |Soft voting, meta-learner, CV stacking — **AUC-ROC: 0.9975**|

### Unsupervised Learning

|Module         |Algorithm              |Key Topics                                                  |
|---------------|-----------------------|------------------------------------------------------------|
|`unsupervised/`|K-Means                |Elbow method, inertia, centroid initialisation              |
|`unsupervised/`|Hierarchical Clustering|Agglomerative, Ward linkage, dendrograms                    |
|`unsupervised/`|DBSCAN                 |Epsilon, min_samples, noise points, arbitrary-shape clusters|

-----

## ⚙️ ML From Scratch

Algorithms implemented without any ML library — only Python and NumPy. The purpose is to verify understanding at the mathematical level before relying on abstracted implementations.

|Notebook                   |What’s Built                                                           |
|---------------------------|-----------------------------------------------------------------------|
|`linear_regression.ipynb`  |Gradient descent, cost function, weight and bias updates               |
|`logistic_regression.ipynb`|Sigmoid activation, binary cross-entropy, manual backpropagation       |
|`knn_regressor.ipynb`      |Euclidean distance computation, k-neighbor aggregation, prediction loop|

-----

## 🚀 Projects

### 🏥 Disease Prediction Pipeline

`projects/disease_prediction_pipeline/`  ·  9,800 patients  ·  Voting Ensemble

A production-ready clinical classification system built for **NovaGen Research Labs**. Classifies patients as healthy or at-risk using a hard-voting ensemble (Logistic Regression + Random Forest + Naïve Bayes) inside a full `sklearn.Pipeline` with stratified cross-validation.

|Metric   |Score                                                |
|---------|-----------------------------------------------------|
|Accuracy |**94.89%**                                           |
|CV Recall|**95.46%**                                           |
|Pipeline |StandardScaler → ColumnTransformer → VotingClassifier|

-----

### 🛒 E-Commerce Purchase Prediction

`projects/ecommerce_purchase_prediction/`  ·  12,330 sessions  ·  Decision Tree

Predicts visitor-to-buyer conversion from behavioral session data. Addresses severe class imbalance (85/15 split) with `class_weight='balanced'` and depth-controlled pruning.

|Metric            |Score     |
|------------------|----------|
|F1-Score          |**0.6485**|
|Baseline benchmark|0.55      |
|Improvement       |**+18%**  |

-----

### 🧬 Thyroid Outlier Detection

`projects/thyroid_outlier_detection/`  ·  1,000 patients  ·  Isolation Forest + LOF

Detects anomalous thyroid hormone profiles using unsupervised anomaly detection — no labeled training data required. Combines Isolation Forest and Local Outlier Factor to flag clinically significant outliers for medical triage and rare disease screening.

|Metric     |Score                                   |
|-----------|----------------------------------------|
|Precision@K|**89%**                                 |
|Method     |Isolation Forest + Local Outlier Factor |
|Use case   |Lab result triage, rare disease flagging|

-----

### 💳 CreditWise Loan Approval System

`projects/creditwise_loan_approval/`  ·  Financial classification  ·  End-to-end

Automated loan risk classification using financial and demographic features. Full preprocessing pipeline with categorical encoding, feature engineering, and scaling. Includes a production `.py` script alongside the exploratory notebook.

-----

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/mehranmushtaq/ml-scikit-scratch.git
cd ml-scikit-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

-----

## 🛠 Tech Stack

|Tool                |Purpose                                             |
|--------------------|----------------------------------------------------|
|Python 3.10+        |Core language                                       |
|Scikit-Learn 1.x    |Algorithms, pipelines, model selection              |
|XGBoost 1.7+        |Gradient boosting                                   |
|Pandas              |Data manipulation and exploration                   |
|NumPy               |Numerical computing and from-scratch implementations|
|Matplotlib / Seaborn|Visualisation and confusion matrices                |
|Jupyter Notebook    |Interactive development                             |

-----

## ✅ Engineering Practices

```
sklearn.Pipeline throughout           — zero data leakage by design
GridSearchCV                          — principled hyperparameter search
Stratified K-Fold Cross-Validation   — reliable generalisation estimates
ColumnTransformer                     — mixed-type feature preprocessing
Class imbalance handling              — class_weight, resampling strategies
Full evaluation suite                 — F1, Precision, Recall, AUC-ROC, Confusion Matrix
Feature importance analysis           — tree-based interpretability
From-scratch implementations          — validates theoretical understanding
random_state seeding                  — fully reproducible results
Production .py scripts                — notebooks converted to deployable code
```

-----

## 🗺 Suggested Learning Path

```
New to ML? Start here and work down:

1. ml_from_scratch/                  → build intuition without abstractions
2. supervised/linear_regression/     → simplest supervised model
3. supervised/logistic_regression/   → move to classification
4. supervised/decision_tree/         → non-linear decision boundaries
5. ensemble_learning/bagging/        → combine models (Random Forest)
6. ensemble_learning/boosting/       → sequential improvement (XGBoost)
7. ensemble_learning/heterogeneous_ensemble/  → mix different model types
8. unsupervised/                     → learn without labels
9. projects/                         → apply everything end-to-end
```

-----

## Author

**Mehran Mushtaq** — Data Science & Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-mehranmushtaq-181717?style=flat-square&logo=github)](https://github.com/mehranmushtaq)

-----



*“First, solve the problem. Then, write the code.”* — John Johnson



⭐ **Star this repo if it helped you** — it keeps the project visible and motivates continued work.


