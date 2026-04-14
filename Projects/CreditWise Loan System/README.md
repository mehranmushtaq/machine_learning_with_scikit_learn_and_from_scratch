# CreditWise Loan System

### *Building an intelligent loan approval engine for SecureTrust Bank*

> **My first end-to-end machine learning project** — a real-world financial ML system built using the complete data science pipeline: data cleaning, EDA, encoding, feature engineering, and multi-model evaluation.

-----

## The Problem

**SecureTrust Bank** is a mid-sized financial company offering personal and home loans to customers across urban and rural regions of **India**. Every day, hundreds of customers apply for loans through online and branch applications.

Until now, the bank has relied on a **manual verification process** — loan officers evaluating applications by checking income proofs, employment details, credit history, and documents. This process is:

-  **Time-consuming** — slows down the approval pipeline
-  **Biased** — subject to individual officer judgment
- **Inconsistent** — same profile, different outcomes

This creates two critical business failures:

> 1. **Good customers sometimes get rejected** — leading to loss of business
> 2. **High-risk customers sometimes get approved** — leading to financial losses

-----

## Problem Statement

SecureTrust Bank wants to introduce an **intelligent loan approval system** powered by Machine Learning that can automatically analyse applicant details and **predict whether a loan should be Approved or Rejected** before final human verification.

**My Role:** Hired as a **Machine Learning Engineer** to design and develop this intelligent system using historical loan application data. The system must:

- Learn hidden patterns from previous customer records
- Provide **accurate, fast, and unbiased** loan approval decisions

-----

## Dataset

Each row represents a **loan applicant** and contains multiple attributes describing their personal, financial, and credit information.

|Property      |Value                       |
|--------------|----------------------------|
|File          |`loan_approval_data.csv`    |
|Samples       |1,000 applicants            |
|Raw Features  |20 columns                  |
|Target        |`Loan_Approved` (Yes / No)  |
|Class Balance |~70% Rejected, ~30% Approved|
|Missing Values|Yes — handled via imputation|

**Features include:**
`Applicant_Income` · `Coapplicant_Income` · `Credit_Score` · `DTI_Ratio` · `Existing_Loans` · `Savings` · `Collateral_Value` · `Loan_Amount` · `Loan_Term` · `Age` · `Dependents` · `Employment_Status` · `Marital_Status` · `Education_Level` · `Gender` · `Property_Area` · `Employer_Category` · `Loan_Purpose`

-----

## Solution Walkthrough

### 1.Data Loading & Cleaning

- Loaded `loan_approval_data.csv` — 1,000 entries, 20 columns
- Identified and handled missing values:
  - **Numerical columns** → imputed with `mean` using `SimpleImputer`
  - **Categorical columns** → imputed with `most_frequent` strategy

### 2.Exploratory Data Analysis (EDA)

**Is the data balanced?**

- 70.2% applications are **rejected** → class imbalance confirmed
- Precision chosen as the primary metric over raw accuracy

**Category Distributions**

- 621 Male · 379 Female applicants
- Graduate vs Not Graduate education split analysed

**Income Distributions**

- `Applicant_Income` and `Coapplicant_Income` are right-skewed → log transform needed

**Feature vs Target Boxplots**

- `Credit_Score` and `DTI_Ratio` showed the clearest separation between approved/rejected groups

**Credit Score Histogram**

- Approved applicants cluster strongly at **700+**
- Rejected applicants spread across 550–700

### 3.Encoding

- **Label Encoding** → `Education_Level`, `Loan_Approved`
- **One-Hot Encoding** → `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`
- Dropped `Applicant_ID` — non-predictive identifier
- Resulting feature set: **28 columns**

### 4.Correlation Heatmap

- Full correlation matrix on all numeric features
- Confirmed `Credit_Score` and `DTI_Ratio` as top predictors

### 5.Train-Test Split & Feature Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### 6.Model Training & Evaluation

#### Logistic Regression

|Metric          |Score                |
|----------------|---------------------|
|Precision       |0.844                |
|Accuracy        |0.807                |
|F1 Score        |0.719                |
|Recall          |0.636                |
|Confusion Matrix|[[209, 14], [30, 77]]|

#### K-Nearest Neighbors (k=5)

|Metric          |Score                |
|----------------|---------------------|
|Precision       |0.727                |
|Accuracy        |0.767                |
|F1 Score        |0.555                |
|Recall          |0.449                |
|Confusion Matrix|[[205, 18], [59, 48]]|

#### Naive Bayes (GaussianNB)

|Metric          |Score                |
|----------------|---------------------|
|Precision       |0.857                |
|Accuracy        |0.858                |
|F1 Score        |0.754                |
|Recall          |0.672                |
|Confusion Matrix|[[211, 12], [35, 72]]|


> **Best Base Model: Naive Bayes** — highest precision (0.857) and accuracy (0.858)

-----

### 7.Feature Engineering

Engineered new features to capture non-linear relationships:

```python
df["DTI_Ratio_sq"]         = df["DTI_Ratio"] ** 2             # amplify high-risk DTI
df["Credit_Score_sq"]      = df["Credit_Score"] ** 2          # reward high credit scores
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"]) # compress income skew
```

**Performance After Feature Engineering:**

|Model              |Precision|Accuracy |F1   |Recall|
|-------------------|---------|---------|-----|------|
|Naive Bayes        |**0.885**|0.879    |0.794|0.720 |
|Logistic Regression|0.856    |**0.885**|0.813|0.776 |

Feature engineering improved Naive Bayes precision from **0.857 → 0.885** (+3.3%)

-----

## Key Findings

|Finding            |Insight                                                                                         |
|-------------------|------------------------------------------------------------------------------------------------|
|Top predictor      |**Credit Score** — highest signal for approval                                                  |
|Second predictor   |**DTI Ratio** — debt burden clearly separates risk classes                                      |
|Best model         |**Naive Bayes** on base precision; **Logistic Regression** competitive after feature engineering|
|Class imbalance    |70/30 split makes precision more business-relevant than accuracy                                |
|Feature engineering|Log + polynomial transforms gave a measurable boost                                             |

-----

## Tech Stack

```
Python 3
pandas · numpy · seaborn · matplotlib
scikit-learn
  ├── Preprocessing  → SimpleImputer, LabelEncoder, OneHotEncoder, StandardScaler
  ├── Models         → LogisticRegression, KNeighborsClassifier, GaussianNB
  └── Evaluation     → confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
```

-----

## Getting Started

```bash
git clone https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch.git
cd "Machine-Learning-with-scikit-learn-and-from-scratch/Projects/CreditWise Loan System"

pip install pandas numpy seaborn matplotlib scikit-learn

jupyter notebook Loan_Approval.ipynb
```

-----

## Lessons Learned

Being my **first ML project**, here’s what building it taught me:

1. **The business problem defines your metric** — SecureTrust needs to minimise bad approvals, so precision matters more than accuracy
1. **Data cleaning is the real work** — imputation choices have downstream consequences
1. **EDA tells you where to look** — the credit score histogram was the single most useful plot
1. **Simple models are powerful** — Naive Bayes outperformed KNN with zero tuning
1. **Feature engineering earns its keep** — 3% precision gain from just three new columns

-----

*Part of the [Machine-Learning-with-scikit-learn-and-from-scratch](https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch) repository.*
