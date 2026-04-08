# Random Forest & Bagging — Titanic Survival Prediction

## Problem Statement

Given passenger data from the Titanic (passenger class, sex, fare, embarkation port, age), predict whether a passenger survived the disaster.

This is a **binary classification** problem (`survived`: 0 = Died, 1 = Survived).

---

## Approach

### 1. Data Preprocessing
- Dataset: Titanic (loaded via Seaborn's built-in library)
- Features used: `pclass`, `sex`, `fare`, `embarked`, `age`
- Missing values filled using median (age) and mode (embarked)
- Categorical variables (`sex`, `embarked`) encoded with `LabelEncoder`
- Train/test split: **70% / 30%** (`random_state=42`)

### 2. Models Trained

| Model | Description |
|---|---|
| Decision Tree | Baseline — max depth 4 to limit overfitting |
| Random Forest | 501 trees, max depth 4, OOB score enabled |
| Bagging (Decision Tree) | 201 unconstrained base decision trees |

### 3. Results

| Model | Accuracy |
|---|---|
| Decision Tree (train) | ~84.75% |
| Decision Tree (test) | ~82.46% |
| Random Forest (OOB) | ~82.02% |
| Random Forest (test) | ~81.72% |
| Bagging DT (test) | ~77.61% |
| Bagging LR (test) | ~79.85% |

---

## Key Concepts

- **OOB Score**: Out-of-bag samples not used by each tree act as an internal validation set — no separate val split needed.
- **Bagging**: Reduces variance by training multiple models on random data subsets and averaging predictions.
- **Random Forest**: Bagging + random feature selection at each split, further decorrelating trees.

---

## Dependencies

```bash
pip install scikit-learn seaborn matplotlib pandas

