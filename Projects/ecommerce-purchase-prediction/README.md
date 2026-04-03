## E-commerce Visitor Purchase Prediction

## Problem Statement
The objective of this project is to act as a Machine Learning Engineer to build a predictive model that determines whether a website visitor is likely to make a purchase (Revenue) based on their session behavior.
Challenges:

• Class Imbalance: The dataset is heavily skewed, with significantly more "non-purchase" sessions than "purchase" sessions (~85% vs 15%).

• Metric Requirement: Due to the imbalance, standard accuracy is misleading. The model must be evaluated using the F1-Score, with a target benchmark of 0.55.

• Model Optimization: The project requires implementing a Decision Tree and utilizing pruning techniques to prevent overfitting and improve generalization.

## Tech Stack
• Language: Python

• Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

• Model: Decision Tree Classifier

## Workflow & Technical Implementation

## 1. Data Preprocessing & EDA
 
• Feature Encoding: Used LabelEncoder to convert categorical variables (Month, VisitorType) and boolean flags (Weekend, Revenue) into numerical formats.

• Feature Selection: Dropped high-cardinality and redundant features such as Browser, OperatingSystems, and Region to reduce noise and focus on high-impact behavioral data.

• Stratified Splitting: Implemented train_test_split with stratify=y to ensure the train and test sets maintained the original distribution of buyers vs. non-buyers.

## 2. Handling Class Imbalance

• Utilized the class_weight='balanced' parameter within the Decision Tree. This penalizes the model more for misclassifying the minority class (buyers), forcing it to learn patterns for successful conversions.

## 3.Model Optimization (Pruning)

To hit the benchmark and avoid a "messy," overfit tree, two pruning methods were implemented:

• Pre-Pruning: Limited max_depth and set min_samples_split to stop the tree from growing too deep.

• Post-Pruning (Cost Complexity Pruning): Applied ccp_alpha to penalize complex branches, resulting in a simpler, more robust model.

## Results

Through strategic pruning and feature selection, the model significantly outperformed the required benchmark:

• Target Benchmark: 0.55 F1-Score

• Achieved F1-Score: 0.6485

• Key Insight: The PageValues feature was identified as the strongest predictor of purchase intent.

## Visualizing the Decision Logic

The following visualization shows the pruned decision tree (max depth 2), demonstrating how the model prioritizes PageValues and Month to segment visitors.
