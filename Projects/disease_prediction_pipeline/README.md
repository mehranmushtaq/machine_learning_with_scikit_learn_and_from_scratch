# Disease Risk Prediction Pipeline

### *Supervised Machine Learning | NovaGen Research Labs*

> **Internship Project** · Biomedical Data Science · Scikit-Learn · Python

-----

## The Story Behind This Project

Imagine being a researcher at **NovaGen Research Labs** — a leading biomedical institute running large-scale population health studies. Every year, thousands of volunteers walk through the doors for medical exams, lifestyle assessments, and clinical tests.

The problem? There’s **no reliable, scalable way** to distinguish who’s genuinely healthy from who’s quietly at risk.

Without this distinction, clinical trial recruitment becomes guesswork. Risk stratification becomes impossible. Research outcomes get muddied.

**That’s where this project comes in.**

-----

## Mission

> *Develop a predictive classification model that determines whether an individual is **healthy** or **unhealthy** — based solely on their health data — to power smarter, data-driven research decisions.*

This directly enables:

- ✅ Selecting eligible participants for clinical trials and longitudinal studies
- ✅ Stratifying populations for risk-based analysis and outcome comparison

-----

## The Dataset

**9,800 unique individuals** · 23 features · 1 binary target

Each record is an independent observation with a rich blend of physiological, lifestyle, and medical history data:

|Feature                |Description                            |
|-----------------------|---------------------------------------|
|`Age`                  |Age of the individual (years)          |
|`BMI`                  |Body Mass Index                        |
|`Blood_Pressure`       |Systolic blood pressure (mmHg)         |
|`Cholesterol`          |Cholesterol level (mg/dL)              |
|`Glucose_Level`        |Blood glucose level (mg/dL)            |
|`Heart_Rate`           |Resting heart rate (bpm)               |
|`Sleep_Hours`          |Average sleep hours per day            |
|`Exercise_Hours`       |Average exercise hours per day         |
|`Water_Intake`         |Daily water intake (litres)            |
|`Stress_Level`         |Stress level (scale 1–10)              |
|`Smoking`              |Smoker (1) / Non-smoker (0)            |
|`Alcohol`              |Alcohol consumption (1/0)              |
|`Diet`                 |Diet category (numerically encoded)    |
|`MentalHealth`         |Mental health score/condition indicator|
|`PhysicalActivity`     |Overall physical activity level        |
|`MedicalHistory`       |Prior medical conditions (1/0)         |
|`Allergies`            |Known allergies (1/0)                  |
|`Diet_Type__Vegan`     |One-hot: Vegan diet                    |
|`Diet_Type__Vegetarian`|One-hot: Vegetarian diet               |
|`Blood_Group_AB/B/O`   |One-hot encoded blood groups           |
|**`Target`**           |**0 = Healthy · 1 = Unhealthy**        |

-----

## Methodology

### 1.Exploratory Data Analysis

- Loaded and inspected the dataset structure
- Encoded categorical columns using `LabelEncoder`
- Built a **Feature Correlation Heatmap** to understand relationships between variables
- Identified that **BMI, Blood Pressure, Cholesterol, and Glucose Level** are the strongest predictors

### 2.Preprocessing

- `StandardScaler` applied within pipelines for all models
- Stratified train/test split (`random_state=42`) to preserve class balance

### 3. Models Trained & Compared

|Model                             |Accuracy  |Mean CV Recall|
|----------------------------------|----------|--------------|
|  Logistic Regression (Polynomial)|87.28%    |88.93%        |
|  Naïve Bayes                     |80.93%    |88.68%        |
|  Random Forest (GridSearchCV)    |**94.89%**|**95.46%**    |
|  Voting Classifier (Ensemble)    |**94.89%**|**95.46%**    |

### 4. Ensemble Strategy

The final model uses a **soft-voting ensemble** combining:

- Polynomial Logistic Regression
- Tuned Random Forest (`max_depth=20`, `n_estimators=300`)
- Naïve Bayes

Weights: `[2, 4, 1]` — favouring the Random Forest’s superior recall performance.

-----

## Results

### Voting Classifier — Classification Report

```
              precision    recall  f1-score   support
           0       0.96      0.93      0.95      1508
           1       0.94      0.97      0.95      1644

    accuracy                           0.95      3152
   macro avg       0.95      0.95      0.95      3152
weighted avg       0.95      0.95      0.95      3152

Voting Accuracy: 94.89%
Cross-Validation Recall Scores: [0.9578, 0.9538, 0.9498, 0.9538, 0.9578]
Mean Recall: 0.9546
```

### Feature Importance (Random Forest)

Top drivers of health outcome prediction:

```
BMI              ████████████████████████  (highest)
Blood_Pressure   ███████████████████
Cholesterol      ████████████████
Glucose_Level    ██████████████
Stress_Level     ████████████
Sleep_Hours      ███████████
Age              ██████████
Water_Intake     █████████
Heart_Rate       ████████
Exercise_Hours   ███████
```

-----

## Project Structure

```
disease_prediction_pipeline/
│
├── disease_prediction_pipeline.ipynb   # Main notebook
├── novagen_dataset.csv                 # Dataset
└── README.md                           
```

-----

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Machine-Learning-with-scikit-learn-and-from-scratch/Projects/disease_prediction_pipeline

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Launch the notebook
jupyter notebook disease_prediction_pipeline.ipynb
```

-----

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

-----

## Key Takeaways

1. **Ensemble methods win** — The Voting Classifier matched Random Forest’s raw accuracy while adding robustness through model diversity.
1. **Physiological markers dominate** — BMI, blood pressure, and cholesterol are far more predictive than lifestyle factors alone.
1. **Recall matters here** — In a health-risk context, missing an unhealthy individual (false negative) is costlier than a false alarm. CV recall of **95.46%** makes this model production-worthy for clinical screening.
1. **Pipeline design is everything** — Wrapping scalers inside pipelines prevents data leakage during cross-validation.

-----

## 👤 Author

**Internship Project — Data Science Track**
Built as part of a supervised ML assignment focused on real-world biomedical classification.

-----

*“The goal is to turn data into information, and information into insight.”* — Carly Fiorina
