## Support Vector Machine (SVM) Implementation

This repository contains practical implementations of Support Vector Machines using scikit-learn. It covers data preprocessing, model training with various kernels, and hyperparameter tuning for both classification and regression tasks.

## Project Overview

Support Vector Machine is a powerful supervised learning algorithm used for both classification and regression. This project demonstrates how to:

• Preprocess data using StandardScaler.

• Implement SVC (Support Vector Classification) using different kernels.

• Optimize models by tuning the regularization parameter **C**.

• (Planned) Implement SVR (Support Vector Regression).

## Tech Stack

• Language: Python

• Libraries: pandas, scikit-learn, numpy

• Environment: Jupyter Notebook 

## Dataset

The classification examples currently use the Iris Dataset, a classic multi-class classification dataset containing:

• 4 Features: Sepal length, Sepal width, Petal length, Petal width.

• Target: 3 classes of iris plants (Setosa, Versicolour, Virginica).

## Implementation Details

## Kernels Explored
   
The notebook compares the performance of different SVM kernels:

• RBF (Radial Basis Function): The default and usually most effective for non-linear data.

• Linear: Best suited for linearly separable data.

• Polynomial: Useful for capturing complex feature interactions.

• Sigmoid: Often used in neural network contexts.

## Performance Comparison

| Kernel | Accuracy | Complexity |
| :--- | :--- | :--- |
| **RBF** | 0.933 | High |
| **Linear** | 0.911 | Low |
| **Sigmoid** | 0.911 | Medium |
| **Polynomial** | 0.866 | Very High |


## Hyperparameter Tuning

A loop was implemented to test different values for the regularization parameter **C**. Higher values of **C** aim for a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly
