## Support Vector Machines (SVM) Implementation

This repository provides a comprehensive implementation of Support Vector Machines (SVM) for both Classification (SVC) and Regression (SVR) tasks. The project focuses on data preprocessing, kernel experimentation, and hyperparameter optimization using scikit-learn.

### Overview

The objective of this project is to demonstrate the versatility of SVMs across different types of data. We explore how different mathematical kernels—Linear, Polynomial, Radial Basis Function (RBF), and Sigmoid—affect the decision boundary and predictive power of the models.


### Technical Workflow

1.	Exploratory Data Analysis (EDA): Checking for null values and understanding feature distributions using pandas.
   
2.	Data Preprocessing:
	
• Feature Scaling: Essential for SVMs, as they are distance-based algorithms. We utilized StandardScaler to ensure all features contribute equally to the distance metric.

• Data Splitting: Used train_test_split with a 70/30 ratio and random_state=42 for reproducibility.

3.	Model Selection & Training:

• Implemented SVC for the Iris Dataset (Classification).

• Implemented SVR for the Diabetes Dataset (Regression).

4.	Hyperparameter Tuning:

• Used GridSearchCV to iterate through combinations of C (regularization), epsilon (margin of error), and kernel types.

### Performance Benchmarks

Classification Results (Iris Dataset)

The model aims to classify flowers into three species. The RBF and Linear kernels showed the highest stability.

| Model/Kernel | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear** | 0.93 | 0.93 | 0.93 | 0.93 |
| **RBF (C=0.5)** | 0.93 | 0.93 | 0.93 | 0.93 |
| **Sigmoid** | 0.91 | 0.92 | 0.91 | 0.91 |
| **Polynomial** | 0.88 | 0.90 | 0.87 | 0.87 |


