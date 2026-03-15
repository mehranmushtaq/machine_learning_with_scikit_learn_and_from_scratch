Titanic Survival Prediction: Decision Tree Analysis
This project demonstrates the end-to-end process of building, visualizing, and optimizing a Decision Tree Classifier to predict passenger survival on the Titanic.
## Key Features
```
• Data Preprocessing: Handling missing values for age and embarked using median and most-frequent imputation.
• Feature Engineering: Categorical encoding of sex and embarked features.
• Hyperparameter Tuning: Systematic iteration through max_depth and min_samples_split to find the optimal balance between bias and variance.
• Cost Complexity Pruning (CCP): Implementation of post-pruning to prevent overfitting by finding the best effective alpha.
• Visualization: High-resolution tree diagrams using matplotlib to interpret model decision logic.
```
## Performance Summary

• Baseline Accuracy: ~76.8% (default parameters).
• Optimized Accuracy: ~82.4% achieved through depth control and pruning.

##  Technologies Used
• Python 3.x
• Scikit-Learn: Model building, preprocessing, and pruning.
• Pandas/NumPy: Data manipulation.
• Matplotlib/Seaborn: Tree visualization and EDA.

## Dataset

The model uses the Seaborn titanic dataset.
• Features: pclass, sex, fare, embarked, age.
• Target: survived.

## Decision Logic
The model identifies Sex as the primary root splitter, followed by Class (pclass) and Age, mirroring historical survival trends where women and children in higher classes had higher priority.
