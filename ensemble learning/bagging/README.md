#  Random Forest & Ensemble Learning — Titanic Survival Prediction

##  Problem Statement

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, the RMS Titanic sank after colliding with an iceberg, killing 1,502 out of 2,224 passengers and crew.

The goal of this project is to build a **binary classification model** that predicts whether a given passenger survived the Titanic disaster, based on personal attributes such as age, sex, ticket class, fare, and port of embarkation.

- **Input (Features):** `pclass`, `sex`, `fare`, `embarked`, `age`
- **Output (Target):** `survived` → `0` = Died, `1` = Survived

This is a classic supervised machine learning problem used to explore and compare ensemble methods against single-model baselines.

---

##  Dataset

- **Source:** Titanic dataset via Seaborn's built-in library (`sns.load_dataset("titanic")`)
- **Size:** 891 passengers
- **Features selected:** passenger class, sex, ticket fare, port of embarkation, age

### Preprocessing Steps
| Step | Method |
|---|---|
| Missing age values | Filled with **median** using `SimpleImputer` |
| Missing embarkation | Filled with **mode** (most frequent) using `SimpleImputer` |
| Categorical encoding | `sex` and `embarked` converted to numeric with `LabelEncoder` |
| Train/Test split | **70% training / 30% testing** with `random_state=42` |

---

##  Models

### 1. Decision Tree (Baseline)
A single Decision Tree trained with `max_depth=4` to prevent extreme overfitting. Acts as the performance baseline.

```python
model = DecisionTreeClassifier(max_depth=4)
```

## 2. Random Forest
An ensemble of 501 decision trees, each trained on a random subset of data and features. Out-of-Bag (OOB) scoring is enabled as a built-in internal validation mechanism — no separate validation split needed.

```
rf = RandomForestClassifier(
    n_estimators=501,
    oob_score=True,
    max_depth=4
)
```

## 3. Bagging Classifier
Uses 201 unconstrained Decision Trees (no max_depth limit) as base learners. Each tree is trained on a random bootstrap sample of the training data. The final prediction is determined by majority vote

```
base_model_dt = DecisionTreeClassifier()
bagging_dt = BaggingClassifier(base_model_dt, n_estimators=201)
```

### 4. Results

| Model | Accuracy |
|---|---|
| Decision Tree (train) | ~84.75% |
| Decision Tree (test) | ~82.46% |
| Random Forest (OOB) | ~82.02% |
| Random Forest (test) | ~81.72% |
| Bagging DT (test) | ~77.61% |
| Bagging LR (test) | ~79.85% |


## Key Concepts

## Bagging (Bootstrap Aggregating)

Trains multiple models independently on different random subsets (with replacement) of the training data, then aggregates predictions. Reduces variance without increasing bias.

## Random Forest

Extends Bagging by also randomly selecting a subset of features at each split. This decorrelates the individual trees, leading to better generalization than plain Bagging.

## Out-of-Bag (OOB) Validation

Because each tree only sees ~63% of the training data, the remaining samples can be used as a free internal validation set. OOB score is a reliable estimate of model performance without needing cross-validation.

## Dependencies

```bash
pip install scikit-learn seaborn matplotlib pandas
