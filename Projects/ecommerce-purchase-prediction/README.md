#  E-Commerce Purchase Prediction

### *Decision Tree Classification | Internship ML Project*

> · Supervised Machine Learning · Scikit-Learn · Python

-----

## The Story Behind This Project

Every second, thousands of people browse online stores — scrolling through product pages, reading reviews, adding items to carts — and then leaving without buying anything.

For an e-commerce business, this is the silent killer of revenue.

The question that keeps product managers and data teams up at night:

> *“Can we predict — before the session ends — whether this visitor will actually make a purchase?”*

If we can answer that with confidence, businesses can intervene in real time: show a discount, trigger a pop-up, personalise a recommendation. The difference between guessing and knowing is worth millions.

**That’s exactly what this project solves.**

-----

## Problem Statement

Given a dataset of **12,330 online shopping sessions**, build a Decision Tree classifier that predicts whether a visitor will generate **Revenue (purchase)** or not — based on their browsing behaviour, session metadata, and visitor profile.

**Primary metric: F1-Score** (not accuracy) — because the dataset is heavily imbalanced (~85% non-buyers), and we care most about correctly identifying the rare but valuable buyers.

>  **Benchmark to beat: F1-Score ≥ 0.55**

-----

## The Dataset — `shop_smart_ecommerce.csv`

**12,330 sessions** · 18 features · 1 binary target (`Revenue`)

|Feature                  |Type  |Description                                  |
|-------------------------|------|---------------------------------------------|
|`Administrative`         |int   |# of admin pages visited                     |
|`Administrative_Duration`|float |Time spent on admin pages                    |
|`Informational`          |int   |# of info pages visited                      |
|`Informational_Duration` |float |Time on info pages                           |
|`ProductRelated`         |int   |# of product pages visited                   |
|`ProductRelated_Duration`|float |Time on product pages                        |
|`BounceRates`            |float |% sessions with single-page visits           |
|`ExitRates`              |float |% exits from a given page                    |
|`PageValues`             |float |Avg value of pages visited before purchase   |
|`SpecialDay`             |float |Closeness to a special day (e.g. Valentine’s)|
|`Month`                  |object|Month of session                             |
|`OperatingSystems`       |int   |Visitor’s OS                                 |
|`Browser`                |int   |Visitor’s browser                            |
|`Region`                 |int   |Geographic region                            |
|`TrafficType`            |int   |Traffic source                               |
|`VisitorType`            |object|Returning / New visitor                      |
|`Weekend`                |bool  |Whether session was on a weekend             |
|**`Revenue`**            |bool  |**Target: True = Purchase**                  |


>  **Class Imbalance**: ~85% of sessions result in no purchase — standard accuracy is misleading here.

-----

## Methodology & Engineering Decisions

### 1.Data Preprocessing

- Loaded `shop_smart_ecommerce.csv` — 12,330 entries, zero nulls
- Applied `LabelEncoder` to categorical/boolean features: `VisitorType`, `Weekend`, `Revenue`, `Month`
- Dropped high-cardinality / low-signal features: `Browser`, `OperatingSystems`, `Region` to reduce noise

### 2.Handling Class Imbalance

- Used `stratify=y` in `train_test_split` to maintain the 85/15 buyer ratio across train and test sets
- Applied `class_weight='balanced'` in Decision Tree — this penalises the model more heavily for misclassifying the minority (buyer) class, forcing it to learn real purchase patterns instead of just predicting “no purchase” every time

### 3.Model Optimisation — Pruning Strategy

An unpruned Decision Tree memorises training noise and collapses on unseen data. Two pruning strategies were implemented to hit the benchmark:

**Pre-Pruning** (`max_depth=2`, `min_samples_split=50`)

- Prevents the tree from growing too deep during training
- Forces the model to focus only on the most discriminative splits

**Post-Pruning** (Cost Complexity Pruning, `ccp_alpha=0.01`)

- Simplifies the already-grown tree by penalising overly complex branches
- Results in a cleaner, more generalisable decision boundary

### 4.Key EDA Insight

A KDE distribution plot of `PageValues` by buyer vs. non-buyer revealed the critical signal:

- Non-buyers are tightly clustered near **PageValues = 0**
- Buyers are spread across **higher PageValues** — they visit more valuable pages before converting

This confirmed `PageValues` as the **root node** in the decision tree.

-----

## Results

### Model Comparison

|Model                 |F1-Score  |Accuracy|Notes                                |
|----------------------|----------|--------|-------------------------------------|
|Baseline Decision Tree|0.5353    |—       |Overfit, no pruning                  |
|Pre-Pruned DT         |**0.6485**|87%     |`max_depth=2`, `min_samples_split=50`|
|Post-Pruned DT (CCP)  |**0.6485**|87%     |`ccp_alpha=0.01`                     |


> **Benchmark smashed: 0.6485 vs target 0.55 (+18% improvement)**

### Pre-Pruned Model — Classification Report

```
              precision    recall  f1-score   support
           0       0.96      0.89      0.92      2084
           1       0.55      0.78      0.65       382

    accuracy                           0.87      2466
   macro avg       0.76      0.83      0.78      2466
weighted avg       0.89      0.87      0.88      2466

Final F1 Score: 0.6485
```

### Key Insight from Decision Tree Visualisation

The pruned tree (max_depth=2) reveals the model’s entire decision logic in just 3 levels:

```
PageValues <= 1.109?
├── True (low value pages) → likely No Purchase
│   ├── Month <= 6.5? → further split
│   └── ProductRelated <= 55.5? → further split
└── False (high value pages) → likely Purchase
    ├── BounceRates <= 0.0? → further split
    └── PageValues <= 23.61? → further split
```

**The #1 predictor of purchase intent: `PageValues`**

-----

## Project Structure

```
ecommerce-purchase-prediction/
│
├── predicting_ecommerce.ipynb    # Main Jupyter notebook
├── shop_smart_ecommerce.csv      # Dataset (12,330 sessions)
└── README.md                     
```

-----

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch

# 2. Navigate to project
cd Projects/ecommerce-purchase-prediction

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 4. Launch notebook
jupyter notebook predicting_ecommerce.ipynb
```

-----

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

-----

## Key Takeaways

1. **F1-Score > Accuracy** — On imbalanced datasets, accuracy is a trap. A model predicting “no purchase” 100% of the time gets 85% accuracy but 0 business value.
1. **PageValues is everything** — One feature dominated the decision tree root across both pruning strategies. Pages with assigned monetary value are visited intentionally by buyers.
1. **Pruning = generalisation** — The unpruned baseline (F1: 0.5353) was worse than both pruned versions (F1: 0.6485), confirming that simpler trees generalise better on real behavioural data.
1. **Stratified splitting matters** — Without `stratify=y`, test sets could underrepresent buyers and make evaluation unreliable.

-----

*“In God we trust. All others must bring data.”* — W. Edwards Deming
