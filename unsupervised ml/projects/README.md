# 🧬 Thyroid Disease Outlier Detection — Unsupervised Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20%7C%20ML-red?style=flat-square)

> Comparative analysis of unsupervised anomaly detection techniques — Isolation Forest, Local Outlier Factor (LOF), and DBSCAN — applied to a real-world thyroid disease dataset to identify clinically significant outliers without labeled anomaly data.

-----

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

-----

## Problem Statement

Thyroid disorders are among the most common endocrine diseases worldwide, affecting millions of patients. Clinical datasets collected from hospitals and diagnostic centers often contain **anomalous patient records** — these may represent:

- Rare or complex disease presentations that deviate from the norm
- Data entry errors or measurement inconsistencies
- Patients with multiple co-occurring conditions
- Edge cases that could mislead predictive models if left unaddressed

Traditional supervised anomaly detection requires **labeled anomaly data**, which is expensive, time-consuming, and often unavailable in healthcare settings. This creates a critical gap: how do we identify unusual patient records **without prior labels**?

This project addresses that gap by applying and comparing three well-established **unsupervised anomaly detection algorithms** to thyroid patient data, enabling automatic flagging of outliers that warrant further clinical or data-quality investigation.

-----

## Objectives

- Apply unsupervised machine learning to detect outliers in a medical dataset
- Compare the performance and behavior of three distinct anomaly detection paradigms
- Visualize high-dimensional outlier patterns using dimensionality reduction (PCA)
- Quantify agreement across methods to assess robustness of detections
- Develop a reusable, well-structured pipeline applicable to similar healthcare datasets

-----

## Dataset

|Property                         |Details                                     |
|---------------------------------|--------------------------------------------|
|**File**                         |`thyroid_dataset.csv`                       |
|**Rows**                         |~6,916 patient records                      |
|**Features**                     |22 columns                                  |
|**Target (for outlier labeling)**|`Outlier_label` (used only to split X and y)|
|**Domain**                       |Endocrinology / Thyroid Disease             |

### Key Features Include:

- `Age`, `Sex` — demographic attributes
- `on_thyroxine`, `query_on_thyroxine` — medication indicators
- `on_antithyroid_medication`, `thyroid_surgery` — treatment history
- `sick`, `pregnant` — clinical condition flags
- `I131_treatment`, `query_hypothyroid` — diagnostic query flags
- `goitre`, `tumor`, `hypopituitary` — pathological indicators

-----

## Methodology

### Pipeline Overview

```
Raw Data
   │
   ▼
Feature/Label Split (X, y)
   │
   ▼
Standard Scaling (StandardScaler)
   │
   ▼
┌──────────────────────────────────────┐
│  Anomaly Detection (3 Methods)       │
│                                      │
│  1. Isolation Forest                 │
│  2. Local Outlier Factor (LOF)       │
│  3. DBSCAN                           │
└──────────────────────────────────────┘
   │
   ▼
PCA (2 Components) for Visualization
   │
   ▼
Comparative Results & Scatter Plots
```

### Algorithms Used

#### 1. Isolation Forest

- **Paradigm:** Tree-based ensemble
- **How it works:** Randomly partitions the feature space; anomalies require fewer splits to isolate
- **Hyperparameters:** `n_estimators=200`, `contamination=0.036`, `random_state=42`
- **Best for:** High-dimensional datasets with global outliers

#### 2.Local Outlier Factor (LOF)

- **Paradigm:** Density-based neighborhood analysis
- **How it works:** Compares the local density of a point to its neighbors; low-density points relative to neighbors are flagged
- **Hyperparameters:** `contamination=0.036`
- **Best for:** Datasets with varying density clusters and local outliers

#### 3.DBSCAN

- **Paradigm:** Clustering-based noise detection
- **How it works:** Groups dense regions into clusters; points that don’t belong to any cluster (noise points) are treated as outliers
- **Hyperparameters:** `eps=2.4`, `min_samples=5`
- **Best for:** Detecting structural outliers and noise across arbitrary cluster shapes

-----

## Results

|Method                  |Outliers Detected|Normal Records|
|------------------------|-----------------|--------------|
|**Isolation Forest**    |249              |6,667         |
|**Local Outlier Factor**|249              |6,667         |
|**DBSCAN**              |257              |6,659         |
|**Dataset Total**       |—                |**6,916**     |

### Key Findings:

- All three methods converge on approximately **249–257 outliers (~3.6% of data)**, demonstrating strong cross-method agreement
- Isolation Forest and LOF produce **identical counts**, suggesting the outliers are robust and not algorithm-specific artifacts
- DBSCAN identifies **8 additional records**, likely capturing density-based anomalies that tree/distance methods miss
- PCA visualizations confirm that outliers consistently appear at the **periphery of the data distribution**

-----

## Visualizations

The notebook produces three PCA-projected scatter plots, one per method, color-coded by outlier label:

|Plot                 |Description                               |
|---------------------|------------------------------------------|
|Isolation Forest Plot|Red = Outlier, Blue = Normal              |
|LOF Plot             |Gradient colormap showing prediction score|
|DBSCAN Plot          |Cluster labels with -1 = Outlier          |

All plots use `matplotlib` with `coolwarm` colormap and are sized at 8×6 inches for clarity.

-----

## Tech Stack

|Tool          |Purpose                            |
|--------------|-----------------------------------|
|`pandas`      |Data loading and manipulation      |
|`numpy`       |Numerical operations               |
|`scikit-learn`|ML algorithms, scaling, PCA        |
|`matplotlib`  |Visualizations                     |
|`seaborn`     |Plot styling                       |
|`JupyterLab`  |Interactive development environment|

-----

## How to Run

### Prerequisites

- Python 3.8+
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Machine-Learning-with-scikit-learn-and-from-scratch/unsupervised_ml/projects/thyroid-outlier-detection.git
cd thyroid-outlier-detection

# 2. Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab

# 3. Launch JupyterLab
jupyter lab

# 4. Open and run thyroid_outlier_detection.ipynb
```

> **Note:** Ensure `thyroid_dataset.csv` is in the same directory as the notebook before running.

-----

## Key Learnings

- Unsupervised anomaly detection is effective even in the **absence of labeled data**, making it highly applicable in real-world healthcare settings
- **Combining multiple detection methods** increases confidence in flagged anomalies — records flagged by all three methods are strong outlier candidates
- **PCA visualization** is essential for interpreting high-dimensional anomaly results intuitively
- Proper **feature scaling** is critical before applying distance-based methods like LOF and DBSCAN
- The `contamination` parameter acts as a domain-knowledge prior — setting it to `0.036` (~3.6%) aligns with typical anomaly rates in clinical data

-----

##  Future Improvements

- [ ] Ensemble voting: flag records identified as outliers by 2+ methods
- [ ] Hyperparameter tuning with grid search for optimal `eps` and `contamination`
- [ ] Evaluate against ground truth labels if available (precision, recall, F1)
- [ ] Deploy as a lightweight Streamlit dashboard for clinical use
- [ ] Extend to other endocrine disease datasets for generalizability

