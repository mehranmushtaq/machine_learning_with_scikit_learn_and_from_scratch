# Heterogeneous Ensemble Methods

> Combining multiple classifier types — Logistic Regression, SVM, and Decision Tree — using Voting and Stacking ensembles on the Breast Cancer dataset.

-----

## Overview

This notebook demonstrates **heterogeneous ensemble learning**, where fundamentally different model architectures are combined to improve predictive performance. Two ensemble strategies are explored:

- **Voting Classifier** — aggregates predictions from multiple base models using soft voting (probability averaging)
- **Stacking Classifier** — trains base models and feeds their outputs to a meta-learner (Logistic Regression) via cross-validation

Both methods are evaluated on scikit-learn’s Breast Cancer dataset (binary classification).

-----

## Results

### Voting Classifier

|Class       |Precision|Recall|F1-Score|Support|
|------------|---------|------|--------|-------|
|0           |0.97     |0.98  |0.98    |63     |
|1           |0.99     |0.98  |0.99    |108    |
|**Accuracy**|         |      |**0.98**|**171**|
|Macro Avg   |0.98     |0.98  |0.98    |171    |
|Weighted Avg|0.98     |0.98  |0.98    |171    |

### Stacking Classifier

|Class       |Precision|Recall|F1-Score|Support|
|------------|---------|------|--------|-------|
|0           |0.97     |0.98  |0.98    |63     |
|1           |0.99     |0.98  |0.99    |108    |
|**Accuracy**|         |      |**0.98**|**171**|
|Macro Avg   |0.98     |0.98  |0.98    |171    |
|Weighted Avg|0.98     |0.98  |0.98    |171    |

**Stacking AUC-ROC Score: 0.9975**

#### Stacking Confusion Matrix

|            |Predicted 0|Predicted 1|
|------------|-----------|-----------|
|**Actual 0**|62         |1          |
|**Actual 1**|2          |106        |

-----

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569 total — 357 malignant (class 1), 212 benign (class 0)
- **Features:** 30 numeric features derived from cell nucleus images
- **Task:** Binary classification
- **Split:** 70% train / 30% test (`random_state=42`)

-----

## Models

### Base Estimators

|Model                   |Details                                                 |
|------------------------|--------------------------------------------------------|
|`LogisticRegression`    |Wrapped in `StandardScaler` pipeline                    |
|`SVC`                   |Wrapped in `StandardScaler` pipeline, `probability=True`|
|`DecisionTreeClassifier`|`max_depth=3`                                           |

### Ensemble Configurations

**Voting:**

```python
VotingClassifier(
    estimators=[("lr", lr_scaled), ("svc", svc_scaled), ("dtc", dtc)],
    voting='soft'  # soft voting uses predicted probabilities
)
```

**Stacking:**

```python
StackingClassifier(
    estimators=[("lr", lr_scaled), ("svc", svc_scaled), ("dtc", dtc)],
    final_estimator=LogisticRegression(),
    cv=5
)
```

-----

## Project Structure

```
heterogeneous_ensemble/
└── heterogeneous_ensemble_methods.ipynb   # Main notebook (253 lines)
```

-----

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install via:

```bash
pip install scikit-learn pandas matplotlib seaborn
```

-----

## Usage

```bash
# Clone the repository
git clone https://github.com/mehranmushtaq/ml-scikit-scratch.git
cd ml-scikit-scratch/ensemble\ learning/heterogeneous_ensemble

# Open the notebook
jupyter notebook heterogeneous_ensemble_methods.ipynb
```

-----

## Key Takeaways

- **Soft voting** (averaging predicted probabilities) generally outperforms hard voting on imbalanced datasets
- **Stacking** leverages cross-validated meta-features to reduce overfitting of the final estimator
- Both ensemble methods achieved **98% accuracy** and near-perfect AUC-ROC (0.9975) on this dataset
- Scaling is critical for distance-based models (LR, SVM) — pipelines ensure no data leakage

-----

## References

- [scikit-learn: VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [scikit-learn: StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)

-----

*Part of the [ml-scikit-scratch](https://github.com/mehranmushtaq/ml-scikit-scratch) repository by [@mehranmushtaq](https://github.com/mehranmushtaq).*
