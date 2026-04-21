# Ensemble Learning: Boosting Methods

A collection of Jupyter notebooks implementing boosting algorithms using scikit-learn and XGBoost.

## Notebooks

### 1. AdaBoost (`ada_boosting.ipynb`)

Adaptive Boosting for binary classification.

- **Base model:** Decision Tree Stump (`max_depth=1`)
- **Estimators:** 100
- **Dataset:** Synthetic classification (500 samples, 20 features, 10 informative)
- **Train/Test split:** 70/30
- **Result:** 76% accuracy, 0.76 F1-score (macro avg)

### 2. Gradient Boosting (`gradient_boosting.ipynb`)

Covers both regression and classification tasks.

**Regressor**

- **Dataset:** Synthetic regression (1000 samples, 10 features, noise=20)
- **Config:** `learning_rate=0.1`, `n_estimators=200`, `max_depth=3`, `subsample=0.8`
- **Result:** R² = 0.919

**Classifier**

- **Dataset:** Synthetic classification (1200 samples, 14 features, 9 informative)
- **Config:** `n_estimators=150`, `learning_rate=0.1`, `max_depth=3`
- **Result:** 90.8% accuracy

### 3. XGBoost (`xgboost.ipynb`)

XGBoost classifier on a synthetic classification task.

- **Dataset:** Synthetic classification (500 samples, 20 features, 10 informative)
- **Config:** `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `eval_metric='logloss'`
- **Result:** 88.9% accuracy, 0.89 F1-score (macro avg)

## Results Summary

|Model            |Task          |Accuracy / R²|
|-----------------|--------------|-------------|
|AdaBoost         |Classification|76%          |
|Gradient Boosting|Regression    |R² = 0.919   |
|Gradient Boosting|Classification|90.8%        |
|XGBoost          |Classification|88.9%        |

## Dependencies

```bash
pip install scikit-learn xgboost numpy
```

## Usage

Open any notebook in Jupyter and run all cells:

```bash
jupyter notebook ada_boosting.ipynb
jupyter notebook gradient_boosting.ipynb
jupyter notebook xgboost.ipynb
```

## References

- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
