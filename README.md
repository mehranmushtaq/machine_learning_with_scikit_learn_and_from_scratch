# 🤖 Machine Learning with Scikit-Learn & From Scratch

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.23+-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)](https://matplotlib.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)

### *End-to-End Machine Learning Repository*

> 🏆 **94.89% accuracy** on Disease Prediction  |  📈 **F1: 0.6485** on E-Commerce (+18% above benchmark)  |  🛒 **4 customer segments** uncovered in SmartCart  |  🧬 **89% Precision@K** on Thyroid Outlier Detection

-----

> **mehranmushtaq** · Internship Learning Track · Python · Scikit-Learn

-----

## What Is This Repository?

This isn’t just a collection of notebooks.

It’s a **complete learning journey** — from implementing algorithms by hand to building production-ready ML pipelines — developed over the course of an intensive internship.

Every folder tells a chapter of that story: understanding *why* an algorithm works before trusting a library to do it. Writing clean pipelines. Tuning models properly. Building real projects that solve real problems — from predicting disease to segmenting customers to flagging medical anomalies.

-----

## What Makes This Different

|Approach                 |What Was Done                                                                    |
|-------------------------|---------------------------------------------------------------------------------|
|**From Scratch**         |Core algorithms implemented in pure Python/NumPy — no sklearn shortcuts          |
|**Pipelines**            |Clean, reproducible workflows using `sklearn.Pipeline` to prevent data leakage   |
|**GridSearchCV**         |Systematic hyperparameter tuning on real datasets                                |
|**Cross-Validation**     |Robust evaluation using k-fold CV with stratification                            |
|**Unsupervised Learning**|Clustering, anomaly detection, and dimensionality reduction applied to real data |
|**End-to-End Projects**  |Full pipelines from raw data → feature engineering → model → evaluation → insight|

-----

##  Projects At a Glance

|Project                                                                                                                                   |Domain    |Technique                    |Key Result                    |
|------------------------------------------------------------------------------------------------------------------------------------------|----------|-----------------------------|------------------------------|
| [SmartCart Customer Segmentation](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/customer-segmentation-smartcart)|Retail    |K-Means + Agglomerative + PCA|4 actionable customer personas|
| [Disease Prediction Pipeline](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/disease_prediction_pipeline)        |Healthcare|Voting Classifier Ensemble   |94.89% accuracy               |
| [E-Commerce Purchase Prediction](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/ecommerce-purchase-prediction)   |Retail    |Decision Tree                |F1: 0.6485 (+18% benchmark)   |
| [CreditWise Loan Approval](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/CreditWise%20Loan%20System)            |Finance   |Classification Pipeline      |End-to-end deployment-ready   |
| [Thyroid Outlier Detection](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/thyroid_outlier_detection)            |Healthcare|Isolation Forest + LOF       |89% Precision@K               |

-----

## 🗂️ Repository Structure

```
ml-scikit-scratch/
│
├── 📁 Datasets/                              # Shared datasets across experiments
│   ├── Emotion_classify_Data.csv
│   ├── Iris.csv
│   ├── Social_Network_Ads.csv
│   ├── house_prices_practice.csv
│   └── insurance.csv
│
├── 📁 linear_regression/
├── 📁 Logistic Regression/
├── 📁 KNN/
├── 📁 Decision tree/
├── 📁 Naive bayes/
├── 📁 Support Vector Machine/
├── 📁 Regularizaton(Lasso:Ridge)/
│
├── 📁 ensemble learning/
│   ├── bagging/                              # Random Forest
│   └── boosting/                             # AdaBoost, Gradient Boosting, XGBoost
│
├── 📁 ml-from-scratch/                       # Pure Python/NumPy — no sklearn
│   ├── linear_reg.ipynb
│   ├── logistic_reg.ipynb
│   └── knn_regressor.ipynb
│
├── 📁 unsupervised ml/
│   ├── k_means.ipynb
│   ├── hiearchichal_clustering.ipynb
│   └── dbscan.ipynb
│
├── 📁 Projects/                              # End-to-end real-world applications
│   ├── 📁 customer-segmentation-smartcart/  
│   ├── 📁 CreditWise Loan System/
│   ├── 📁 ecommerce-purchase-prediction/
│   ├── 📁 thyroid_outlier_detection/
│   └── 📁 disease_prediction_pipeline/
│
├── notebook_vs_production.md
├── requirements.txt
└── README.md
```

-----

## What This Repository Covers

### Supervised Learning

- **[Linear Regression](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/linear_regression)** — Predicting continuous outcomes; from OLS to regularised versions
- **[Logistic Regression](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Logistic%20Regression)** — Binary and multi-class classification with polynomial features
- **[K-Nearest Neighbours (KNN)](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/KNN)** — Distance-based classification and regression
- **[Decision Trees](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Decision%20tree)** — Classifier & regressor with pre/post pruning strategies
- **[Naive Bayes](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Naive%20bayes)** — Probabilistic classification with Gaussian distributions
- **[Support Vector Machines](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Support%20Vector%20Machine)** — SVC for classification, SVR for regression
- **[Regularization](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Regularizaton(Lasso%3ARidge))** — Lasso (L1) and Ridge (L2) to control overfitting

### Ensemble Methods

- **[Bagging](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/ensemble%20learning/bagging)** — Random Forest with feature importance analysis
- **[Boosting](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/ensemble%20learning/boosting)** — AdaBoost, Gradient Boosting, and XGBoost

### Unsupervised Learning

- **[K-Means Clustering](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/unsupervised%20ml)** — Centroid-based segmentation with elbow & silhouette analysis
- **[Hierarchical Clustering](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/unsupervised%20ml)** — Agglomerative dendrograms with Ward linkage
- **[DBSCAN](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/unsupervised%20ml)** — Density-based, anomaly-resistant clustering

### ML From Scratch (Pure Python / NumPy)

- **[Linear Regression](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/ml-from-scratch)** — Gradient descent, cost function, weight updates
- **[Logistic Regression](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/ml-from-scratch)** — Sigmoid, binary cross-entropy, manual backprop
- **[KNN Regressor](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/ml-from-scratch)** — Euclidean distance, neighbor voting

-----

## Featured Projects

-----

###  [SmartCart Customer Segmentation](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/customer-segmentation-smartcart)  `NEW`

> **Domain:** Retail  ·  **Type:** Unsupervised Learning  ·  **Dataset:** 2,240 customers × 22 features

**The Problem:** SmartCart had thousands of customers but was treating them all the same — one message, one offer, one strategy. The goal was to discover natural groupings in the customer base so that marketing, product, and retention teams could act with precision.

**The Challenge:** No labels, no ground truth. The model had to find structure in raw behavioral and demographic data entirely on its own, and the number of segments wasn’t known upfront.

**Approach & Key Decisions:**

- **Missing values:** 24 records had missing `Income` — imputed using the **median** rather than the mean to avoid distortion from high-earner outliers
- **Feature engineering:** Raw columns were transformed into richer signals — `Age` (from birth year), `Customer_Tenure_Days` (loyalty duration), `Total_Spending` (sum of all category spend), `Total_Children`, and simplified `Education` and `Living_With` groups
- **Outlier removal:** Customers aged 90+ and income above $600k were dropped, reducing the dataset from 2,240 → 2,236 without losing signal
- **Encoding & scaling:** `OneHotEncoder` for categorical features, `StandardScaler` across all — both done inside a clean pipeline to prevent leakage
- **Dimensionality reduction:** PCA reduced the encoded feature space to **4 principal components** (~55% variance explained: 23.2%, 11.4%, 10.4%, 9.9%) — essential for meaningful distance-based clustering
- **K selection:** Used both the **Elbow Method (WCSS)** and **Silhouette Score** in a combined plot — both converged on **k = 4**
- **Final model:** Agglomerative Clustering with Ward linkage was selected over K-Means for its deterministic results and tighter, more balanced cluster boundaries

**Features Used:** `Income`, `Recency`, `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`, `Complain`, `Response`, `Age`, `Customer_Tenure_Days`, `Total_Spending`, `Total_Children`, `Education` (encoded), `Living_With` (encoded)

**Results — Discovered Customer Segments:**

|Cluster|Avg Income|Avg Spending|Avg Age|Avg Children|Profile                          |
|-------|----------|------------|-------|------------|---------------------------------|
|0      |~$42,706  |~$327       |~56 yrs|1.23        |Budget-conscious, larger families|
|1      |~$66,279  |~$1,055     |~59 yrs|0.62        |Affluent loyalists, catalog-heavy|
|2      |~$35,326  |~$110       |~55 yrs|1.25        |Low-income, high web visits      |
|3      |~$74,727  |~$1,271     |~59 yrs|~0.37       |Premium spenders, fewest children|

**Business Impact:**

|Segment                            |Recommended Strategy                                                        |
|-----------------------------------|----------------------------------------------------------------------------|
|Cluster 0 — Budget Families        |Bundle deals, multi-buy discounts, family-oriented promotions               |
|Cluster 1 — Affluent Loyalists     |Catalog upsells, premium loyalty rewards, cross-category nudges             |
|Cluster 2 — Low-Engagement Browsers|Re-engagement email campaigns, web-exclusive offers, abandoned cart recovery|
|Cluster 3 — Premium High-Spenders  |VIP early access, luxury product launches, concierge-style service          |

**Tech:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn` (PCA, OneHotEncoder, StandardScaler, KMeans, AgglomerativeClustering, silhouette_score), `kneed`

-----

### [Disease Prediction Pipeline](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/disease_prediction_pipeline)

> **Domain:** Healthcare  ·  **Type:** Supervised Classification  ·  **Dataset:** 9,800 patients (NovaGen Research Labs)

**The Problem:** NovaGen Research Labs needed a reliable automated system to classify patients as healthy or at risk based on clinical features — to streamline participant selection for clinical trials and reduce manual screening time.

**The Challenge:** Healthcare data demands high recall — missing a sick patient is far more costly than a false alarm. A single model risked being too sensitive to noise. The solution required an ensemble that stayed robust across diverse patient profiles.

**Approach & Key Decisions:**

- **Pipeline-first design:** The entire workflow — imputation, encoding, scaling, and modeling — was encapsulated inside a single `sklearn.Pipeline` to guarantee zero data leakage between train and test sets
- **Ensemble strategy:** A **hard-voting classifier** combining Logistic Regression, Random Forest, and Naïve Bayes — each contributing a vote, with the majority deciding the final prediction. This reduces the variance of any single model’s weaknesses
- **Stratified K-Fold CV:** Used stratified splits throughout to ensure class proportions were preserved across every fold, critical for a medical dataset
- **Evaluation focus:** Optimized for **recall** alongside accuracy — in clinical settings, false negatives are the higher-risk failure mode

**Results:**

|Metric             |Score                              |
|-------------------|-----------------------------------|
|Accuracy           |**94.89%**                         |
|CV Recall          |**95.46%**                         |
|Validation Strategy|Stratified K-Fold Cross-Validation |
|Model              |Hard-Voting Ensemble (LR + RF + NB)|

**Real-World Use Case:** Automated pre-screening of 9,800 patient records for clinical trial eligibility — reducing manual review workload by flagging candidates with 95%+ recall.

**Tech:** `pandas`, `sklearn` (Pipeline, VotingClassifier, LogisticRegression, RandomForestClassifier, GaussianNB, StratifiedKFold, cross_val_score)

-----

### [E-Commerce Purchase Prediction](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/ecommerce-purchase-prediction)

> **Domain:** Retail  ·  **Type:** Supervised Classification  ·  **Dataset:** 12,330 browsing sessions (ShopSmart)

**The Problem:** ShopSmart had 12,330 recorded browsing sessions but only ~15% ended in a purchase. The marketing team needed a model to identify which visitors were likely to convert — so they could target interventions at the right moment.

**The Challenge:** An extreme **85/15 class imbalance** meant that a naïve model could score 85% accuracy by predicting “no purchase” every time — completely useless in practice. The real target was maximising F1-Score, not accuracy.

**Approach & Key Decisions:**

- **Class imbalance handling:** Used `class_weight='balanced'` in the Decision Tree — this automatically adjusts sample weights inversely proportional to class frequency, forcing the model to pay attention to the minority (purchase) class
- **Depth pruning:** `max_depth` tuning via GridSearchCV to prevent overfitting on the majority class and improve generalisation to purchase patterns
- **Feature set:** Session-level behavioral signals — page views, visit duration, bounce rate, exit rate, special day proximity, month, visitor type, and browser/region/OS metadata
- **Evaluation:** Prioritised **F1-Score** as the primary metric — the harmonic mean of precision and recall, which penalises models that sacrifice one for the other

**Results:**

|Metric         |Score                               |
|---------------|------------------------------------|
|F1-Score       |**0.6485**                          |
|Benchmark F1   |0.55                                |
|Improvement    |**+18% above benchmark**            |
|Class Imbalance|85% non-purchase / 15% purchase     |
|Validation     |Train/test split with stratification|

**Real-World Use Case:** Flag high-intent sessions in real time — trigger discount popups, live chat, or cart reminders for users the model identifies as likely converters.

**Tech:** `pandas`, `sklearn` (DecisionTreeClassifier, GridSearchCV, classification_report, train_test_split), `matplotlib`, `seaborn`

-----

### [CreditWise Loan Approval System](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/CreditWise%20Loan%20System)

> **Domain:** Finance  ·  **Type:** Supervised Classification  ·  **Deliverable:** Notebook + Production `.py` script

**The Problem:** Manual loan approval is slow, inconsistent, and prone to bias. CreditWise needed an automated classification system that could assess applicant risk from financial and demographic data — consistently, at scale.

**The Challenge:** Financial data is messy — mixed types, missing values, skewed distributions, and categorical variables that need careful encoding. The pipeline had to handle all of this cleanly before any model could be trained.

**Approach & Key Decisions:**

- **Feature engineering:** Derived new risk signals from raw inputs — debt-to-income proxies, employment stability indicators, and interaction terms between credit history and income band
- **Preprocessing pipeline:** `ColumnTransformer` applied different strategies to numerical (imputation + scaling) and categorical (encoding) features simultaneously — keeping the workflow clean and reproducible
- **Model selection:** Multiple classifiers evaluated — Logistic Regression, Random Forest, and Gradient Boosting — with `GridSearchCV` used to tune the best-performing model
- **Production delivery:** The final model was exported as a standalone `.py` script alongside the notebook, making it directly deployable to a backend service or batch scoring system

**Highlights:**

|Aspect        |Detail                                                                    |
|--------------|--------------------------------------------------------------------------|
|Input features|Income, loan amount, credit history, employment, dependents, property area|
|Preprocessing |Median imputation, OneHot encoding, StandardScaler                        |
|Evaluation    |Accuracy, Precision, Recall, F1, Confusion Matrix                         |
|Deliverable   |Notebook (exploration) + `.py` script (production-ready)                  |

**Real-World Use Case:** Automated loan pre-screening — score thousands of applications in seconds, surface high-risk rejections and fast-track low-risk approvals.

**Tech:** `pandas`, `sklearn` (Pipeline, ColumnTransformer, GridSearchCV, RandomForestClassifier, GradientBoostingClassifier, LogisticRegression), `matplotlib`

-----

### [Thyroid Outlier Detection](https://github.com/mehranmushtaq/ml-scikit-scratch/tree/main/Projects/thyroid_outlier_detection)

> **Domain:** Healthcare  ·  **Type:** Unsupervised Anomaly Detection  ·  **Dataset:** 1,000 patient lab records

**The Problem:** In a dataset of 1,000 thyroid hormone lab results, a small number of patients had profiles that were clinically abnormal — but there were no labels indicating which ones. The system had to find anomalies without any ground truth.

**The Challenge:** Unsupervised anomaly detection is difficult to evaluate — there’s no label to train on, and different methods can disagree on what counts as “abnormal.” Combining multiple detectors and validating against clinical thresholds was essential.

**Approach & Key Decisions:**

- **Dual-detector approach:** Both **Isolation Forest** and **Local Outlier Factor (LOF)** were applied independently — Isolation Forest excels at global anomalies (extreme values), while LOF catches local outliers (unusual relative to nearby points). Agreement between both detectors increased confidence in flagged cases
- **No labels required:** The system learns the “normal” distribution from the data itself — any patient whose profile deviates significantly from the learned structure is flagged
- **Feature set:** Thyroid hormone levels (T3, T4, TSH) and related clinical markers — standardised before detection to prevent scale bias
- **Threshold tuning:** Contamination parameter tuned to reflect clinical expectations of outlier prevalence in the patient population
- **Evaluation:** Since no ground truth labels existed, results were validated by checking flagged patients against known clinical reference ranges for hypo/hyperthyroid conditions

**Results:**

|Metric         |Score                    |
|---------------|-------------------------|
|Precision@K    |**89%**                  |
|Detectors Used |Isolation Forest + LOF   |
|Dataset        |1,000 patient lab records|
|Labels Required|None (fully unsupervised)|

**Real-World Use Case:** Automated lab result triage — flag patients for clinician review before they present with symptoms, enabling earlier intervention for thyroid disorders.

**Tech:** `pandas`, `numpy`, `sklearn` (IsolationForest, LocalOutlierFactor, StandardScaler), `matplotlib`, `seaborn`

-----

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/mehranmushtaq/ml-scikit-scratch.git

# Navigate into the repo
cd ml-scikit-scratch

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

-----

## Key Concepts Practiced

```
  Data Preprocessing & Feature Engineering
  Encoding (LabelEncoder, One-Hot, ColumnTransformer)
  Train/Test Split with Stratification
  StandardScaler inside Pipelines (zero data leakage)
  GridSearchCV for Hyperparameter Tuning
  Cross-Validation (k-fold, stratified)
  Class Imbalance Handling (class_weight, resampling)
  Dimensionality Reduction via PCA
  Unsupervised Validation (Elbow Method, Silhouette Score)
  Anomaly Detection (Isolation Forest, LOF)
  Model Evaluation (Accuracy, F1, Precision, Recall, AUC)
  Confusion Matrix Analysis
  Feature Importance Visualisation
  Algorithms Implemented From Scratch (NumPy only)
```

-----

## Author

**Mehran Mushtaq** · Data Science & Machine Learning Track

[![GitHub](https://img.shields.io/badge/GitHub-mehranmushtaq-181717?style=flat-square&logo=github)](https://github.com/mehranmushtaq)

-----

*“First, solve the problem. Then, write the code.”* — John Johnson


