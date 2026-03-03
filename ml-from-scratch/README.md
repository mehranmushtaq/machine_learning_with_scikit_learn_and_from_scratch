## Machine Learning Algorithms From Scratch

This directory contains **Machine Learning algorithms implemented from scratch using NumPy only**, without using high-level libraries like scikit-learn.

The goal of this project is to deeply understand the **mathematics and inner workings of Machine Learning algorithms.**

## Algorithms Implemented

## Linear Regression (Gradient Descent)

File: linear_reg.ipynb

Linear Regression models the relationship between input features **X** and target values **y**.

## Implementation includes:
	•	Gradient Descent Optimization
	•	Mean Squared Error Loss
	•	Weight and Bias Updates
	•	Multi-feature support

Model Equation:

```
y = wX + b
```

## Features:

✔ Gradient Descent optimization

✔ Bias and weights calculation

✔ Prediction function

✔ Vectorized NumPy operations

## Key Concepts:

	•	Loss minimization
	•	Parameter updates
	•	Convergence
  
## Ordinary Least Squares (OLS Method)

File: linear_reg.ipynb

OLS finds the best-fit line analytically by minimizing squared error.

## Implementation includes:
	Ordinary Least Squares
	Normal Equation solution

OLS Formula:

```
β = (XᵀX)^(-1)Xᵀy
```

## Features:

✔ Analytical solution

✔ No iterations required

✔ Fast computation

✔ Matrix operations

## Key Concepts:
	•	Exact solution
	•	Matrix algebra
	•	No iterations required

## Logistic Regression

File: logistic_reg.ipynb

## Implementation includes:
	•	Gradient Descent Optimization
	•	Sigmoid Function
	•	Binary Classification
	•	Probability Prediction

Sigmoid Function:
```
σ(z) = 1 / (1 + e^(-z))
```
## Key Concepts:
	•	Classification
	•	Decision Boundary
	•	Probability estimation

## K-Nearest Neighbors (KNN) – Regressor

File: knn_regressor.ipynb

This implementation builds a KNN Regressor from scratch using NumPy only, without relying on scikit-learn.

Unlike parametric models (Linear Regression), KNN is a:

	•	Non-parametric model
	•	Instance-based learning algorithm
	•	Lazy learner (no explicit training phase)


How It Works

	1.	Store training data
	2.	Compute Euclidean distance from test point to all training points
	3.	Select K nearest neighbors
	4.	Return the mean of their target values
## Distance Formula:
```
d(x, y) = √ Σ (xᵢ - yᵢ)²
```

## Features Implemented

	•	Adjustable K value
	•	Euclidean distance metric
	•	Efficient NumPy-based computation
	•	Mean-based regression prediction
	•	Supports multi-dimensional input

## Key Concepts Learned
	•	Distance metrics
	•	Bias-variance tradeoff (effect of K)
	•	Instance-based learning
	•	Local averaging behavior
	•	Model complexity control via K

## Key Concepts Learned
	•	Distance metrics
	•	Bias-variance tradeoff (effect of K)
	•	Instance-based learning
	•	Local averaging behavior
	•	Model complexity control via K


## ⚠️ Important Insight

	•	Small K → Low bias, High variance
	•	Large K → High bias, Low variance

Choosing K properly is crucial.


## What I Learned

Through implementing these algorithms from scratch, I learned:

## Linear Regression
	•	How Gradient Descent updates parameters
	•	How learning rate affects convergence
	•	The role of bias and weights
	•	Difference between iterative solution and closed-form (OLS)
	•	Matrix multiplication in ML

## Logistic Regression
	•	How sigmoid transforms linear output into probability
	•	How classification differs from regression
	•	Decision boundary concept
	•	Binary cross-entropy intuition
	•	Why gradient descent is required for classification

## KNN (Regressor)
	•	Distance-based learning
	•	Instance-based (lazy) learning
	•	Bias–variance tradeoff via K
	•	Why KNN has no training phase
	•	How local averaging works

## Mathematical Foundations Strengthened
	•	Vectorization with NumPy
	•	Dot products & matrix algebra
	•	Optimization intuition
	•	How loss functions drive learning
	•	Why scaling matters
	
## Engineering Skills
	•	Writing ML models as classes
	•	Clean separation of fit() and predict()
	•	Reusable code structure
	•	Avoiding library shortcuts

