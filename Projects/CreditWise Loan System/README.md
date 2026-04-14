#  CreditWise Loan System

### *Can a machine learn to trust a borrower?*

> **My first end-to-end machine learning project** — built to solve a real-world financial problem using the complete ML pipeline: data cleaning, EDA, feature engineering, and multi-model evaluation.

----

##  The Story

Every day, banks reject thousands of loan applications — and approve thousands they shouldn’t. A loan officer reviews income, credit score, employment status, and a dozen other signals to make a binary decision: **approve or deny**.

But human judgment is slow, inconsistent, and sometimes biased.

**The question I set out to answer:**

> *Can we train a machine learning model to predict loan approval decisions accurately — and perhaps more fairly — than manual review?*

This project tackles that question using a dataset of 1,000 loan applicants, walking through the full data science workflow from raw CSV to a tuned, evaluated classifier.

-----

##  Problem Statement

**Goal:** Build a binary classification model that predicts whether a loan application will be **approved (Yes)** or **rejected (No)** based on applicant financial and demographic features.

**Why it matters:**

- Faster loan processing for applicants
- Consistent, data-driven decisions for lenders
- Identifying the key financial signals that drive creditworthiness

**Challenge:** The dataset is imbalanced — ~70% of loans are rejected — making raw accuracy a misleading metric. Precision becomes the north star.

-----

## Dataset

|Property      |Value                       |
|--------------|----------------------------|
|File          |`loan_approval_data.csv`    |
|Samples       |1,000 applicants            |
|Features      |20 columns                  |
|Target        |`Loan_Approved` (Yes / No)  |
|Missing Values|Yes — handled via imputation|

**Key Features:**
`Applicant_Income` · `Coapplicant_Income` · `Credit_Score` · `DTI_Ratio` · `Existing_Loans` · `Savings` · `Collateral_Value` · `Loan_Amount` · `Loan_Term` · `Age` · `Dependents` · `Employment_Status` · `Marital_Status` · `Education_Level` · `Gender` · `Property_Area` · `Employer_Category` · `Loan_Purpose`

-----

##  Project Walkthrough

### 1.Data Loading & Cleaning

- Loaded dataset with `pandas`, inspected shape and dtypes
- Identified missing values across numerical and categorical columns
- **Numerical columns** → imputed with `mean` using `SimpleImputer`
- **Categorical columns** → imputed with `most_frequent` strategy

### 2.Exploratory Data Analysis (EDA)

**Class Balance Check**

- ~70.2% of applications are **rejected** (imbalanced dataset)
- Pie chart confirms the need to look beyond accuracy

**Category Distributions**

- 621 Male vs 379 Female applicants
- Graduate vs Not Graduate split explored

**Income Distributions**

- Histograms of `Applicant_Income` and `Coapplicant_Income` reveal right-skewed distributions

**Feature vs Target (Boxplots)**
Compared `Applicant_Income`, `Credit_Score`, `DTI_Ratio`, `Savings`, `Age`, and `Loan_Amount` across approved/rejected groups — credit score and DTI ratio showed the clearest separation.

**Credit Score Histogram**

- Approved loans cluster at higher credit scores (700+)
- Rejected loans more spread across the 550–700 range

### 3.Encoding

- **Label Encoding** → `Education_Level`, `Loan_Approved`
- **One-Hot Encoding** → `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`
- Dropped `Applicant_ID` (non-informative)
- Final encoded dataframe: **28 columns**

### 4.Correlation Heatmap

- Generated full correlation matrix on numeric features
- Identified multicollinearity candidates
- `Credit_Score` and `DTI_Ratio` showed strong signals toward the target

### 5.Train-Test Split & Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### 6.Model Training & Evaluation

Three classifiers were trained and evaluated on **Precision, Accuracy, F1, Recall, and Confusion Matrix**:

#### Logistic Regression

|Metric          |Score                |
|----------------|---------------------|
|Precision       |0.844                |
|Accuracy        |0.807                |
|F1 Score        |0.719                |
|Recall Score    |0.636                |
|Confusion Matrix|[[209, 14], [30, 77]]|

#### K-Nearest Neighbors (k=5)

|Metric          |Score                |
|----------------|---------------------|
|Precision       |0.727                |
|Accuracy        |0.767                |
|F1 Score        |0.555                |
|Recall Score    |0.449                |
|Confusion Matrix|[[205, 18], [59, 48]]|

#### Naive Bayes (GaussianNB)

|Metric          |Score                |
|----------------|---------------------|
|Precision       |0.857                |
|Accuracy        |0.858                |
|F1 Score        |0.754                |
|Recall Score    |0.672                |
|Confusion Matrix|[[211, 12], [35, 72]]|


> **Best Model: Naive Bayes** — highest precision (0.857) and accuracy (0.858)

-----

### 7.Feature Engineering

To push performance further, new features were engineered:

```python
df["DTI_Ratio_sq"]         = df["DTI_Ratio"] ** 2             # non-linear DTI signal
df["Credit_Score_sq"]      = df["Credit_Score"] ** 2          # amplify credit score signal
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"]) # normalize skewed income
```

**Results after Feature Engineering — Naive Bayes:**

|Metric      |Before|After    |
|------------|------|---------|
|Precision   |0.857 |**0.885**|
|Accuracy    |0.858 |**0.879**|
|F1 Score    |0.754 |0.794    |
|Recall Score|0.672 |0.720    |

**Results after Feature Engineering — Logistic Regression:**

|Metric      |Score|
|------------|-----|
|Precision   |0.856|
|Accuracy    |0.885|
|F1 Score    |0.813|
|Recall Score|0.776|

-----

## Key Findings

- **Credit Score** is the single strongest predictor of loan approval
- **DTI Ratio** (Debt-to-Income) is the second most predictive feature
- **Naive Bayes** outperformed KNN and matched Logistic Regression in base precision
- **Feature engineering** (log transform + polynomial features) gave a measurable boost
- The class imbalance (~70/30) means precision is more meaningful than raw accuracy

-----

## Tech Stack

```
Python 3
pandas · numpy · seaborn · matplotlib
scikit-learn
  └── SimpleImputer · LabelEncoder · OneHotEncoder
  └── train_test_split · StandardScaler
  └── LogisticRegression · KNeighborsClassifier · GaussianNB
  └── confusion_matrix · accuracy_score · precision_score · f1_score · recall_score
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

This being my **first ML project**, here’s what I took away:

1. **Data cleaning takes longer than modeling** — imputation decisions matter more than you think
1. **Never trust accuracy on imbalanced data** — always check precision and recall together
1. **Simple models can surprise you** — Naive Bayes beat KNN by a significant margin
1. **Feature engineering has real impact** — squaring DTI and log-transforming income improved precision by ~3%
1. **EDA is not optional** — the boxplots and histograms told a story that drove every modeling decision

-----

## Future Improvements

- [ ] Try Random Forest and XGBoost for comparison
- [ ] Apply SMOTE to handle class imbalance
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Build a Streamlit app for interactive loan prediction
- [ ] Cross-validation instead of a single train/test split

-----

*Part of the [Machine-Learning-with-scikit-learn-and-from-scratch](https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch) repository.*
