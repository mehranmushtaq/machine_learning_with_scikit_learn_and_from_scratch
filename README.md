# Machine Learning with Scikit-Learn & From Scratch

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

### *End-to-End Machine Learning Repository*

> 🏆 **94.89% accuracy** on Disease Prediction  |  📈 **F1: 0.6485** on E-Commerce (18% above benchmark)  |  🔬 **9,800+ patient** dataset  |  🧬 **89% Precision@K** on Thyroid Outlier Detection

-----

> **mehranmushtaq**
> · Internship Learning Track
> · Python
> · Scikit-Learn

-----

## What Is This Repository?

This isn’t just a collection of notebooks.

It’s a **complete learning journey** — from implementing algorithms by hand to deploying production-ready ML pipelines — built over the course of an intensive internship.

Every folder tells a chapter of that story: understanding *why* an algorithm works before trusting a library to do it. Writing clean pipelines. Tuning models properly. Building real projects that solve real problems.

-----

## What Makes This Different

|Approach               |What Was Done                                                          |
|-----------------------|-----------------------------------------------------------------------|
|**From Scratch**       |Core algorithms implemented in pure Python/NumPy — no sklearn shortcuts|
|**Pipelines**          |Clean, reproducible workflows using `sklearn.Pipeline`                 |
|**GridSearchCV**       |Systematic hyperparameter tuning on real datasets                      |
|**Cross-Validation**   |Robust evaluation using k-fold CV to prevent data leakage              |
|**End-to-End Projects**|Full pipelines from raw data → model → evaluation → insight            |

-----

## Repository Structure

```
ml-scikit-scratch/
│
├── Datasets/                              # Shared datasets across experiments
│   ├── Emotion_classify_Data.csv
│   ├── Iris.csv
│   ├── Social_Network_Ads.csv
│   ├── house_prices_practice.csv
│   └── insurance.csv
│
├── Decision Tree/
│   ├── decision_tree_classifier.ipynb
│   ├── decision_tree_regressor.ipynb
│   └── README.md
│
├── KNN/
│   ├── Knn.ipynb
│   └── README.md
│
├── linear_regression/
│   ├── Linear_regression.ipynb
│   └── README.md
│
├── Logistic Regression/
│   ├── Logistic_Regressor.ipynb
│   └── README.md
│
├── Naive Bayes/
│   ├── naive_bayes.ipynb
│   └── README.md
│
├── Regularization(Lasso/Ridge)/
│   ├── lasso_ridge.ipynb
│   └── README.md
│
├── Support Vector Machine/
│   ├── svc.ipynb
│   ├── svr.ipynb
│   └── README.md
│
├── ensemble learning/
│   ├── bagging/
│   │   ├── Random_forest.ipynb
│   │   └── README.md
│   └── boosting/
│       ├── gradient_boosting.ipynb
│       ├── ada_boosting.ipynb
│       ├── xgboost.ipynb
│       └── README.md
│
├── ml-from-scratch/                       # Algorithms without sklearn
│   ├── linear_reg.ipynb
│   ├── logistic_reg.ipynb
│   └── knn_regressor.ipynb
│
├── unsupervised ml/
│   ├── dbscan.ipynb
│   ├── hiearchichal_clustering.ipynb
│   ├── k_means.ipynb
│   ├── k_means_clustering.ipynb
│   └── README.md
│
├── Projects/                              # Applied end-to-end projects
│   ├── CreditWise Loan System/
│   │   ├── loan_Approval.ipynb
│   │   ├── loan_approval.py
│   │   ├── loan_approval_data.csv
│   │   └── README.md
│   ├── ecommerce-purchase-prediction/
│   │   ├── predicting_ecommerce.ipynb
│   │   ├── predicting_ecommerce.py
│   │   ├── shop_smart_ecommerce.csv
│   │   └── README.md
│   ├── thyroid_outlier_detection/
│   │   ├── thyroid_outlier_detection.ipynb
│   │   ├── thyroid_dataset.csv
│   │   └── README.md
│   └── disease_prediction_pipeline/
│       ├── disease_prediction_pipeline.ipynb
│       ├── novagen_dataset.csv
│       └── README.md
│
├── notebooks_vs_production.md
├── requirements.txt
└── README.md
```

-----

## What This Repository Covers

### Supervised Learning

- **Linear Regression** — Predicting continuous outcomes; from OLS to regularised versions
- **Logistic Regression** — Binary and multi-class classification with polynomial features
- **K-Nearest Neighbours (KNN)** — Distance-based classification and regression
- **Decision Trees** — Classifier & regressor with pre/post pruning strategies
- **Naive Bayes** — Probabilistic classification with Gaussian distributions
- **Support Vector Machines** — SVC for classification, SVR for regression
- **Regularization** — Lasso (L1) and Ridge (L2) to control overfitting

### Ensemble Methods

- **Bagging** — Random Forest with feature importance analysis
- **Boosting** — AdaBoost, Gradient Boosting, and XGBoost

### Unsupervised Learning

- **K-Means Clustering** — Centroid-based segmentation
- **Hierarchical Clustering** — Agglomerative dendrograms
- **DBSCAN** — Density-based anomaly-resistant clustering

### ML From Scratch

- Linear Regression (pure NumPy)
- Logistic Regression (gradient descent by hand)
- KNN Regressor (distance metrics implemented manually)

-----

## ⭐ Featured Projects

### 🏥 [Disease Prediction Pipeline](https://github.com/mehranmushtaq/ml-scikit-scratch/blob/main/Projects/disease_prediction_pipeline)

> *NovaGen Research Labs · 9,800 patients · Voting Classifier Ensemble*

|Metric     |Score     |
|-----------|----------|
|✅ Accuracy |**94.89%**|
|✅ CV Recall|**95.46%**|

Classifies individuals as healthy or unhealthy using a full ensemble pipeline (Logistic Regression + Random Forest + Naïve Bayes). Production-ready for clinical trial participant selection.

-----

### 🛒 [E-Commerce Purchase Prediction](https://github.com/mehranmushtaq/ml-scikit-scratch/blob/main/Projects/ecommerce-purchase-prediction)

> *12,330 sessions · Decision Tree · F1-Score benchmark: 0.55*

|Metric           |Score     |
|-----------------|----------|
|✅ F1-Score       |**0.6485**|
|✅ Above Benchmark|**+18%**  |

Predicts whether an online visitor will convert to a buyer. Tackled heavy class imbalance (85/15 split) with `class_weight='balanced'` and pruning strategies.

-----

### 🧬 [Thyroid Outlier Detection](https://github.com/mehranmushtaq/ml-scikit-scratch/blob/main/Projects/thyroid_outlier_detection)

> *Unsupervised Anomaly Detection · Isolation Forest + LOF · 1,000 patients*

|Metric       |Score                     |
|-------------|--------------------------|
|✅ Precision@K|**89%**                   |
|✅ Method     |**Isolation Forest + LOF**|

Detects anomalous thyroid hormone profiles in patient lab data using unsupervised anomaly detection — no labeled training data required. Flags clinically significant outliers for medical triage and rare disease screening.

-----

### 💳 [CreditWise Loan Approval System](https://github.com/mehranmushtaq/ml-scikit-scratch/blob/main/Projects/CreditWise%20Loan%20System)

> *Loan risk classification · End-to-end pipeline*

Automated loan approval prediction using financial and demographic features. Full preprocessing pipeline with feature engineering and model evaluation.

-----

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/mehranmushtaq/ml-scikit-scratch.git

# Navigate into the repo
cd ml-scikit-scratch

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

-----

## Key Concepts Practiced

```
  Data Preprocessing & Feature Engineering
  Encoding (LabelEncoder, One-Hot)
  Train/Test Split with Stratification
  StandardScaler inside Pipelines (no data leakage)
  GridSearchCV for Hyperparameter Tuning
  Cross-Validation (k-fold, stratified)
  Class Imbalance Handling
  Model Evaluation (Accuracy, F1, Precision, Recall, AUC)
  Confusion Matrix Analysis
  Feature Importance Visualisation
  Algorithms Implemented From Scratch
```

-----

## Author

**Mehran Mushtaq** · Data Science & Machine Learning Track

[![GitHub](https://img.shields.io/badge/GitHub-mehranmushtaq-181717?style=flat-square&logo=github)](https://github.com/mehranmushtaq)

-----

*“First, solve the problem. Then, write the code.”* — John Johnson
