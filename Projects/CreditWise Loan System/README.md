# CreditWise Loan System

### *Building an intelligent loan approval engine for SecureTrust Bank*

> **My first end-to-end machine learning project** — a real-world financial ML system built using the complete data science pipeline: data cleaning, EDA, encoding, feature engineering, and multi-model evaluation.

-----

## The Problem

**SecureTrust Bank** is a mid-sized financial company offering personal and home loans to customers across urban and rural regions of **India**. Every day, hundreds of customers apply for loans through online and branch applications.

Until now, the bank has relied on a **manual verification process** — loan officers evaluating applications by checking income proofs, employment details, credit history, and documents. This process is:

- **Time-consuming** — slows down the approval pipeline
- **Biased** — subject to individual officer judgment
- **Inconsistent** — same profile, different outcomes

This creates two critical business failures:

> 1. **Good customers sometimes get rejected** — leading to loss of business
> 1. **High-risk customers sometimes get approved** — leading to financial losses

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

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

-----

## Getting Started

### Option 1: Run the Python Script (Standalone)

The project includes a complete Python script (`loan_approval.py`) that you can run directly from your terminal.

#### Prerequisites

Ensure you have **Python 3.7+** installed. Check with:

```bash
python --version
```

#### Installation & Setup

1. **Clone the repository:**
   
   ```bash
   git clone https://github.com/mehranmushtaq/machine_learning_with_scikit_learn_and_from_scratch.git
   cd "machine_learning_with_scikit_learn_and_from_scratch/Projects/CreditWise Loan System"
   ```
1. **Install required dependencies:**
   
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
1. **Ensure the dataset is in the same directory:**
- The script expects `loan_approval_data.csv` to be in the same folder as `loan_approval.py`

#### Running the Script

Execute the script from your terminal:

```bash
python loan_approval.py
```

**What happens when you run it:**

- Loads and cleans the loan approval dataset
- Performs Exploratory Data Analysis (EDA) with visualizations
- Encodes categorical features
- Trains three ML models: Logistic Regression, K-Nearest Neighbors, and Naive Bayes
- Displays evaluation metrics (Precision, Accuracy, F1 Score, Recall, Confusion Matrix)
- Performs feature engineering on the best model
- Shows performance comparison before and after feature engineering
- Displays all generated plots (histograms, boxplots, heatmaps, etc.)

#### Expected Output

The terminal will print:

- Data shape and missing values summary
- Class distribution
- Model performance metrics for each algorithm
- Feature importance rankings
- Visualization plots will pop up in a window

-----

### Option 2: Run the Jupyter Notebook

Alternatively, you can explore the analysis interactively using Jupyter:

```bash
jupyter notebook Loan_Approval.ipynb
```

-----

## File Structure

```
CreditWise Loan System/
│
├── loan_approval.py              # ✅ Standalone Python script (Run this from terminal!)
├── Loan_Approval.ipynb           # Jupyter notebook with full analysis
├── loan_approval_data.csv        # Dataset (required for script to run)
└── README.md                     # This file
```

-----

## Troubleshooting

**Error: `No module named 'pandas'`**

- Solution: Run `pip install pandas numpy seaborn matplotlib scikit-learn`

**Error: `loan_approval_data.csv not found`**

- Solution: Make sure the CSV file is in the same directory as `loan_approval.py`

**Error: `ModuleNotFoundError` for seaborn or matplotlib**

- Solution: Install missing packages with `pip install seaborn matplotlib`

**Plots not displaying**

- The script uses `plt.show()` at the end to display all plots. Make sure your terminal/IDE supports GUI windows, or run in an environment with display capabilities.

-----

## Lessons Learned

Being my **first ML project**, here’s what building it taught me:

1. **The business problem defines your metric** — SecureTrust needs to minimise bad approvals, so precision matters more than accuracy
1. **Data cleaning is the real work** — imputation choices have downstream consequences
1. **EDA tells you where to look** — the credit score histogram was the single most useful plot
1. **Simple models are powerful** — Naive Bayes outperformed KNN with zero tuning
1. **Feature engineering earns its keep** — 3% precision gain from just three new columns

-----

## Next Steps

To improve this project further, consider:

- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Cross-validation for more robust evaluation
- Handling class imbalance (SMOTE, class weights)
- Ensemble methods (Random Forest, Gradient Boosting)
- Model deployment as a REST API

-----

*Part of the [Machine-Learning-with-scikit-learn-and-from-scratch](https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch) repository.*
