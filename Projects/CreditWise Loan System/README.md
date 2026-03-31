## CreditWise: Automated Loan Approval Engine

## Executive Summary
CreditWise is a high-performance machine learning system designed to automate the credit risk assessment process. By leveraging historical applicant data—including debt-to-income (DTI) ratios, credit scores, and collateral values—the system predicts loan approval viability with high precision.
The primary objective is to reduce manual underwriting overhead while maintaining a strict risk profile to minimize defaults.

## Technical Architecture
The project implements a robust data preprocessing pipeline to ensure production-grade reliability and prevent data leakage.

## Exploratory Data Analysis (EDA)

• Target Imbalance: Analyzed class distribution (approx. 70/30 split) to ensure model robustness.

• Feature Correlation: Identifed high-impact variables such as Credit_Score (0.45 correlation) and DTI_Ratio (-0.44 correlation).

• Visual Insights: Utilized segmented boxplots and distribution histograms to identify non-linear relationships in financial metrics.

## Feature Engineering & Pipeline

To optimize the performance of the Gaussian Naive Bayes and Logistic Regression models, the following transformations were applied:

• Logarithmic Scaling: Applied to Applicant_Income to normalize skewed distributions.

• Polynomial Features: Created squared terms for Credit_Score and DTI_Ratio to capture non-linear decision boundaries.

• Standardization: Implemented StandardScaler to ensure feature parity for gradient-based solvers.

• Categorical Encoding: Strategic use of OneHotEncoder (dropping first to avoid multicollinearity) and LabelEncoder for ordinal features.

## Model Performance Metrics
After rigorous cross-validation and testing (33% holdout), the models achieved the following results:

| Model               | Precision | Accuracy | F1 Score | Recall |
|:--------------------|:---------:|:---------:|:--------:|:------:|
| Naive Bayes         | 0.885     | 0.878     | 0.793    | 0.719  |
| Logistic Regression | 0.855     | 0.884     | 0.813    | 0.775  |

Naive Bayes was selected as the champion model for this deployment due to its superior Precision (0.885). In a lending context, high precision is critical as it minimizes "False Positives"—approving a loan for a high-risk candidate who is likely to default.

## Deployment & Usage

The system is architected for easy integration into banking web portals using Streamlit.

## Prerequisties
``
pip install pandas numpy scikit-learn matplotlib seaborn joblib
``

