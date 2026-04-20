import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

def load_and_preprocess(filepath):
    """Loads data and handles encoding and feature selection."""
    data = pd.read_csv(filepath)
    
    # Initialize LabelEncoder for categorical/boolean features
    le = LabelEncoder()
    categorical_cols = ["VisitorType", "Weekend", "Revenue", "Month"]
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    # Feature Selection: Dropping redundant/high-cardinality features
    X = data.drop(['Revenue', 'Browser', 'OperatingSystems', 'Region'], axis=1)
    y = data['Revenue']
    
    # Stratify ensures the 85/15 class imbalance is maintained
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), X.columns

def run_models(X_train, X_test, y_train, y_test, feature_names):
    """Runs all three models found in the notebook."""
    
    # 1. Basic Decision Tree
    print("\n--- 1. Basic Decision Tree ---")
    dt_basic = DecisionTreeClassifier()
    dt_basic.fit(X_train, y_train)
    print(f"F1 Score: {f1_score(y_test, dt_basic.predict(X_test)):.4f}")

    # 2. Pre-Pruned Decision Tree (Your best performing model)
    print("\n--- 2. Pre-Pruned Decision Tree ---")
    dt_pruned = DecisionTreeClassifier(
        max_depth=2, 
        min_samples_split=50, 
        class_weight='balanced', 
        random_state=42
    )
    dt_pruned.fit(X_train, y_train)
    y_pred_pruned = dt_pruned.predict(X_test)
    print(f"Final F1 Score: {f1_score(y_test, y_pred_pruned):.4f}")
    print(classification_report(y_test, y_pred_pruned))

    # 3. Logistic Regression
    print("\n--- 3. Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=40000, class_weight='balanced', solver="liblinear")
    lr_model.fit(X_train, y_train)
    print(f"F1 Score: {f1_score(y_test, lr_model.predict(X_test)):.4f}")

    return dt_pruned

def main():
    file_path = "shop_smart_ecommerce.csv"
    
    try:
        # 1. Data Prep
        (X_train, X_test, y_train, y_test), feat_names = load_and_preprocess(file_path)
        
        # 2. Model Execution
        best_model = run_models(X_train, X_test, y_train, y_test, feat_names)
        
        # 3. Save Visualization (Optional but helpful for .py files)
        plt.figure(figsize=(20, 10))
        plot_tree(best_model, filled=True, feature_names=feat_names, 
                  class_names=['No Purchase', 'Purchase'], max_depth=2)
        plt.savefig('decision_tree_viz.png')
        print("\nVisualization saved as 'decision_tree_viz.png'")

    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it's in the same folder.")

if __name__ == "__main__":
    main()