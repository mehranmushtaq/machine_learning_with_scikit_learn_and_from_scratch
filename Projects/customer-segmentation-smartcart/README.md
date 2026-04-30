#  SmartCart Customer Segmentation

> **Unsupervised machine learning pipeline for intelligent customer profiling using K-Means and Agglomerative Clustering with PCA dimensionality reduction.**

-----

##  Overview

This project builds an end-to-end customer segmentation system for SmartCart, a retail platform, using behavioral and demographic data. By identifying distinct customer groups, the business can deliver personalized marketing campaigns, optimize product recommendations, and improve customer lifetime value.

The pipeline covers the full ML lifecycle — from raw data ingestion and preprocessing through feature engineering, dimensionality reduction, clustering, and strategic cluster interpretation.

-----

##  Dataset

**File:** `smartcart_customers.csv`  
**Shape:** 2,240 rows × 22 columns (post-cleaning: 2,236 × 15)

|Category         |Features                                                                                               |
|-----------------|-------------------------------------------------------------------------------------------------------|
|Demographics     |`Year_Birth`, `Education`, `Marital_Status`, `Income`                                                  |
|Household        |`Kidhome`, `Teenhome`                                                                                  |
|Purchase Behavior|`MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`      |
|Channel Activity |`NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`|
|Engagement       |`Recency`, `Response`, `Complain`, `Dt_Customer`                                                       |

-----

##  Pipeline Architecture

```
Raw Data
   │
   ▼
Data Preprocessing
   ├── Handle Missing Values (Income → median imputation)
   └── Null check across all 22 columns
   │
   ▼
Feature Engineering
   ├── Age               = 2026 − Year_Birth
   ├── Customer_Tenure_Days = days since enrollment
   ├── Total_Spending    = sum of all Mnt* columns
   ├── Total_Children    = Kidhome + Teenhome
   ├── Education         → [Undergraduate, Graduate, Postgraduate]
   └── Living_With       → [Partner, Alone]
   │
   ▼
Drop Columns
   └── Remove raw/redundant columns (ID, Year_Birth, Dt_Customer, etc.)
   │
   ▼
Outlier Removal
   ├── Age < 90
   └── Income < 600,000
   │
   ▼
Feature Encoding & Scaling
   ├── OneHotEncoder  → Education, Living_With
   └── StandardScaler → All features
   │
   ▼
Dimensionality Reduction
   └── PCA (4 components) — explains ~55% of variance
       [0.232, 0.114, 0.104, 0.099]
   │
   ▼
Optimal K Selection
   ├── Elbow Method (WCSS)
   └── Silhouette Score
   │
   ▼
Clustering
   ├── K-Means Clustering       (k=4, random_state=42)
   └── Agglomerative Clustering (k=4, linkage=ward)
   │
   ▼
Cluster Characterization & Interpretation
```

-----

## Key Results

### Optimal K = 4

Both the Elbow Method and Silhouette Score analysis converged on **4 clusters** as the optimal segmentation.

### Cluster Summary (Agglomerative — Final Model)

|Cluster|Avg Income|Avg Total Spending|Avg Age|Profile                                    |
|-------|----------|------------------|-------|-------------------------------------------|
|**0**  |~$42,706  |~$327             |~56 yrs|Budget-conscious, older, larger families   |
|**1**  |~$66,279  |~$1,055           |~59 yrs|High-income, high-spend, catalog buyers    |
|**2**  |~$35,326  |~$110             |~55 yrs|Low-income, minimal spend, high web visits |
|**3**  |~$74,727  |~$1,271           |~59 yrs|Premium segment — highest income & spending|

-----

##  Tech Stack

|Tool                    |Purpose                                                                |
|------------------------|-----------------------------------------------------------------------|
|`pandas` / `numpy`      |Data manipulation                                                      |
|`matplotlib` / `seaborn`|Visualization                                                          |
|`scikit-learn`          |Preprocessing, PCA, K-Means, Agglomerative Clustering, Silhouette Score|
|`kneed`                 |Automated elbow detection (KneeLocator)                                |

-----

##  Project Structure

```
customer-segmentation-smartcart/
│
├── smartcart_customer_segmentation.ipynb   # Main notebook
├── smartcart_customers.csv                 # Raw dataset
└── README.md                               # Project documentation
```

-----

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

### Run the Notebook

```bash
git clone https://github.com/mehranmushtaq/ml-scikit-scratch.git
cd ml-scikit-scratch/Projects/customer-segmentation-smartcart
jupyter notebook smartcart_customer_segmentation.ipynb
```

-----

##  Methodology Notes

- **Missing values** in `Income` (24 records) were imputed using the **median** to avoid skew from outliers.
- **PCA** was applied after scaling to reduce the encoded feature space to 4 principal components before clustering, improving cluster quality and enabling 3D visualization.
- **Agglomerative Clustering with Ward linkage** was selected as the final model due to its deterministic nature and better-defined cluster boundaries compared to K-Means.
- Cluster distributions were validated via count plots and Income vs. Total Spending scatter analysis.

-----

##  Business Applications

|Segment                            |Recommended Strategy                       |
|-----------------------------------|-------------------------------------------|
|Cluster 0 — Budget Families        |Discount-driven promotions, bundle deals   |
|Cluster 1 — Affluent Loyalists     |Premium loyalty rewards, catalog upsells   |
|Cluster 2 — Low-Engagement Browsers|Re-engagement campaigns, web-first offers  |
|Cluster 3 — Premium High-Spenders  |VIP programs, early access, luxury products|

-----

##  Author

**Mehran Mushtaq**  
[GitHub → mehranmushtaq/ml-scikit-scratch](https://github.com/mehranmushtaq/ml-scikit-scratch)

-----

*Built as part of a hands-on ML portfolio series covering end-to-end scikit-learn pipelines from scratch.*
