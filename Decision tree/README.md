# Decision Tree Classification: Titanic Survival

This directory contains the implementation of a **Decision Tree Classifier** to predict passenger survival on the Titanic. The focus here is on understanding how a single tree splits data and how to prevent **overfitting** through various pruning techniques.

## Decision Tree Regression: Diabetes Progression

This directory contains the implementation of a Decision Tree Regressor to predict a quantitative measure of disease progression in diabetes patients. The focus here is on understanding how trees handle continuous numerical targets and how to control variance through hyperparameter constraints.

##  Contents
* **`decision_tree_classifier.ipynb`**: Step-by-step implementation, visualization, and pruning.
* **`decision_tree_regressor.ipynb`**: Step-by-step implementation, including feature selection and model evaluation.
* **`README.md`**: Explanation of Decision Tree concepts and results.

## Concepts Implemented

### 1. Pre-Pruning (Early Stopping)
To prevent the tree from growing too deep and memorizing noise in the data, we experiment with:
* **Max Depth**: Limiting how many levels the tree can have.
* **Min Samples Split**: The minimum number of samples required to split an internal node.
* **Observation**: As seen in the code, increasing depth initially improves accuracy but eventually leads to overfitting on the training set.
* **Min Samples Leaf**: The minimum number of samples required to be at a leaf node.
* **Observation**: Unlike classification where we measure accuracy, in regression, these constraints directly impact the Mean Squared Error (MSE) and R2 score.

### 2. Post-Pruning (Cost Complexity Pruning)
We use **Minimal Cost Complexity Pruning** to find the optimal tree size.
* **ccp_alpha**: This parameter controls the trade-off between the tree's complexity and its accuracy. 
* By calculating the effective alphas for the tree, we can find the "weakest" links and prune them to ensure the model generalizes well to new data.

### 3. Evaluation Metrics for Regression
Since we are predicting a continuous value rather than a category, we evaluate the model using:
* **Mean Squared Error (MSE)**: Measures the average squared difference between the estimated values and the actual value.
* **R2 Score (Coefficient of Determination):** Represents the proportion of variance for the target variable... Our test $R2$ achieved approximately 0.41.


### 3. Visualization
Using `plot_tree`, we visualize the decision-making process.
* **Root Node**: Typically splits on `Sex` or `Pclass`, which are the strongest predictors.
* **Leaf Nodes**: Show the final prediction (Died vs. Survived) based on the path taken.
* **Feature Split**: The criteria used to divide the data (e.g., s5 <= 0.014).
* **Squared Error**: The impurity measure for that specific node.
* **Value**: The predicted numerical value for samples falling into that branch


##  Results Summary
| Technique | Best Parameter | Test Accuracy (%) |
| :--- | :--- | :--- |
| Pre-Pruning | `max_depth=4` | ~82.46% |
| Post-Pruning | `ccp_alpha ≈ 0.0017` | ~82.08% |
| Default (Unpruned) | N/A | ~76.86% |

##  How to use
1. Run **`decision_tree_classifier.ipynb`/`decision_tree_regressor.ipynb`** to view the impact of different pruning methods.
2. Observe the accuracy improvements as we move from a default tree to a pruned one.







