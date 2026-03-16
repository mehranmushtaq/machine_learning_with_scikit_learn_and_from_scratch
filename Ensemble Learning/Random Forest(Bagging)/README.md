# Ensemble Learning: Random Forest & Bagging

This directory explores **Ensemble Learning** techniques, focusing on how combining multiple "weak" learners (like Decision Trees) can create a "strong" learner with higher accuracy and better generalization.

##  Contents

* `Random_forest.ipynb`: Implementation of Random Forest and Bagging Classifiers using the Titanic dataset.
* `README.md`: Overview of ensemble concepts and results.

##  Concepts Implemented

### 1. Bagging (Bootstrap Aggregating)
Bagging reduces the variance of an estimate by taking many samples from the training set, building a predictor for each sample, and averaging the predictions. 

* **Base Models used:** Decision Trees.

### 2. Random Forest
An extension of Bagging that provides even more diversity. In addition to drawing random samples of data, it also selects a **random subset of features** at each split in the tree. This decorrelates the trees, making the model more robust than a single Decision Tree.



### 3. Hyperparameter Tuning

Used `GridSearchCV` to optimize the following parameters:
* `n_estimators`: Number of trees in the forest.
* `max_depth`: Maximum depth of the trees to prevent overfitting.
* `criterion`: The function to measure the quality of a split (Gini vs. Entropy).


## Results Summary

| Model | Accuracy (%) |
| :--- | :--- |
| Single Decision Tree | ~82.4% |
| Random Forest (Tuned) | ~81.8% (OOB Score) |
| Bagging (Logistic Reg) | ~79.5% |


##  How to use
1. Ensure the `Datasets/` folder contains the Titanic data or use the Seaborn built-in loader.
2. Install dependencies: `pip install -r ../requirements.txt`.
3. Run `Random_forest.ipynb` to see the comparison and tuning process.
