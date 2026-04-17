#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("loan_approval_data.csv")
df.head()
df.info()

# # Handle Mising Values 
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

from sklearn.impute import SimpleImputer
num_imp = SimpleImputer(strategy = "mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
cat_imp = SimpleImputer(strategy = "most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])


# EDA- exploratory Data Analysis
# how balanced our classes are?
classes_count = df["Loan_Approved"].value_counts()
plt.pie(classes_count,labels=["No","Yes"],autopct="%1.1f%%")
plt.title("Is Loan Approved or Not?")

#Categories
gender_cnt = df["Gender"].value_counts()
ax = sns.barplot(gender_cnt)
ax.bar_label(ax.containers[0])

edu_cnt = df["Education_Level"].value_counts()
ax = sns.barplot(edu_cnt)
ax.bar_label(ax.containers[0])

fig,axes = plt.subplots(1,2)
sns.histplot(ax = axes [0],data=df, x = "Applicant_Income",bins=20)
sns.histplot(ax = axes [1],data=df, x = "Coapplicant_Income",bins=20)
plt.tight_layout()


fig,axes = plt.subplots(2,3)
sns.boxplot(ax = axes [0,0],data = df,x = "Loan_Approved",y = "Applicant_Income")
sns.boxplot(ax = axes [0,1],data = df,x = "Loan_Approved",y = "Credit_Score")
sns.boxplot(ax = axes [0,2],data = df,x = "Loan_Approved",y = "DTI_Ratio")
sns.boxplot(ax = axes [1,0],data = df,x = "Loan_Approved",y = "Savings")
sns.boxplot(ax = axes [1,1],data = df,x = "Loan_Approved",y = "Age")
sns.boxplot(ax = axes [1,2],data = df,x = "Loan_Approved",y = "Loan_Amount")
plt.tight_layout()



#Credit Score With Loan Approved
sns.histplot(
    data=df, 
    x="Credit_Score",
    hue="Loan_Approved",
    bins=20,
    multiple="dodge"
)

#Remove Applicant Id
df = df.drop("Applicant_ID",axis=1)

# Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# LabelEncoder
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

# OneHotEncoder
cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
ohe = OneHotEncoder(drop="first",sparse_output= False, handle_unknown ="ignore")
encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded,columns=ohe.get_feature_names_out(cols),index=df.index)
df = pd.concat([df.drop(columns=cols),encoded_df],axis=1)
df.head()


# Correlation Heatmap
num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols.corr()

plt.figure(figsize=(15,8))
sns.heatmap(
    corr_matrix,
    annot = True,
    fmt=".2f",
    cmap = "coolwarm"
)
num_cols.corr()["Loan_Approved"].sort_values(ascending=False)

# Train-Test-Split + Feature Scaling
X = df.drop("Loan_Approved",axis=1)
y = df["Loan_Approved"]

X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

X_train.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

 # Train And Evaluate Models

# Logistic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
log_model = LogisticRegression()
log_model.fit(X_train_scaled,y_train)
y_pred = log_model.predict(X_test_scaled)

#Evaluation
print("Precision:",precision_score(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))
print("f1 Score:",f1_score(y_test,y_pred))
print("Recall Score:",recall_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model =  KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(X_train_scaled,y_train)
y_pred = knn_model.predict(X_test_scaled)

#Evaluation
print("Precision:",precision_score(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))
print("f1 Score:",f1_score(y_test,y_pred))
print("Recall Score:",recall_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_model =  GaussianNB()
nb_model.fit(X_train_scaled,y_train)
y_pred = nb_model.predict(X_test_scaled)

#Evaluation
print("Precision:",precision_score(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))
print("f1 Score:",f1_score(y_test,y_pred))
print("Recall Score:",recall_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

# # Best Model on the basis pf Precision => Naive Bayes
#  Feature Engineering
# Add Or Transform Feature
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])
X = df.drop(columns=["DTI_Ratio","Credit_Score","Loan_Approved","Applicant_Income"])
y = df["Loan_Approved"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes
nb_model =  GaussianNB()
nb_model.fit(X_train_scaled,y_train)
y_pred = nb_model.predict(X_test_scaled)

#Evaluation
print("Precision:",precision_score(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))
print("f1 Score:",f1_score(y_test,y_pred))
print("Recall Score:",recall_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled,y_train)
y_pred = log_model.predict(X_test_scaled)

#Evaluation
print("Precision:",precision_score(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))
print("f1 Score:",f1_score(y_test,y_pred))
print("Recall Score:",recall_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

# Evaluation for Logistic Regression
print("\n--- Logistic Regression Metrics ---")
print("Precision:" , precision_score(y_test, y_pred))
print("Accuracy:" , accuracy_score(y_test, y_pred))
print("f1 Score:" , f1_score(y_test, y_pred))
print("Recall Score:" , recall_score(y_test, y_pred))
print("Confusion Matrix:\n" , confusion_matrix(y_test, y_pred))

# Final step to display any plots generated earlier in the script
import matplotlib.pyplot as plt
plt.show()
