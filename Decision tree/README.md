## Titanic Survival Prediction using Decision Tree

## Overview

This project predicts whether a passenger survived the Titanic disaster using a Decision Tree Classifier implemented with Scikit-learn. It also demonstrates pre-pruning and post-pruning techniques to reduce overfitting.

## Dataset

Dataset used: Seaborn Titanic dataset

## Target variable:

	•	survived (0 = No, 1 = Yes)

## Features used:

	•	pclass
	•	sex
	•	age
	•	fare
	•	embarked

## Data Preprocessing
	•	Missing values handled using SimpleImputer
	•	Categorical variables encoded using LabelEncoder
	•	Dataset split using train_test_split

## Model

## Algorithm used:
	•	DecisionTreeClassifier

## Experiments
	1.	Baseline Decision Tree
	2.	Pre-Pruning (max_depth, min_samples_split)
	3.	Post-Pruning using Cost Complexity Pruning (ccp_alpha