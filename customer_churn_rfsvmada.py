#!/usr/bin/env python
# coding: utf-8

# # Predicting Customer Churn Using Machine Learning

# ## Introduction & Problem Statement  
# 
# This project focuses on binary classification  to predict customer churn (Yes/No) for a telecom company using historical customer data. The goal is to identify at-risk customers early to implement retention strategies. This means that when rating performance, it will be more important to catch potential customer churn that it will be to incorrectly predict customer churn for a customer not at risk.
# 
# This is a supervised learning classification task.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


# ## Data Source & Description  
# 
# **Dataset** : The data is sourced from Kaggle Telecom Customer Churn Dataset .
# Size : 7043 rows × 21 columns|
# 
# **Feature Fields**: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
# 
# **Target Field**: Churn

# In[2]:


# Load dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Display first few rows
df.head()


# ## Exploratory Data Analysis (EDA)  
# 
#  - Class Imbalance : Visualized distribution of "Churn" (1035 "No" vs. 374 "Yes") using a bar chart.
#  - Feature Relationships :
#    - Heatmap : Checked collinearity (e.g., "MonthlyCharges" and "TotalCharges" showed moderate correlation, but tree-based models handle this well).
#    - Feature Importance : Preliminary insights from decision trees suggested "MonthlyCharges" and "Contract" as critical predictors.
#          
#      

# In[3]:


# Summary statistics
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Distribution of target variable ('Churn')
plt.figure(figsize=(8, 4))
sns.countplot(x='Churn', data=df)
plt.title('Class Distribution (Churn)')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 6))
corr_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Example: Tenure vs Churn
plt.figure(figsize=(10, 5))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Customer Tenure by Churn')
plt.show()


# ## Data Cleaning  
# 
# Steps Taken :   
# 
# 1. Handling Missing Values : TotalCharges has 11 null values. These were dropped
# 1. Data Type Fixes : Converted categorical variables (e.g., "SeniorCitizen" from numeric to categorical).
# 1. Data Type Fixes : Convert TotalCharges to a numeric value
# 1. Feature Removal : Dropped redundant columns like "customerID" (no predictive value).
# 1. Feature Scaling : Used StandardScaler to standardize numerical features.
#      

# In[4]:


# Convert categorical variables to numerical (using one-hot encoding)
cat_df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 
                                'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                'StreamingTV', 'StreamingMovies', 'Contract', 
                                'PaperlessBilling', 'PaymentMethod'], drop_first=True)

cat_df['TotalCharges'] = pd.to_numeric(cat_df['TotalCharges'], errors='coerce')

print(f'# of null values for TotalCharges: {cat_df['TotalCharges'].isnull().sum()}')
cat_df.dropna(subset=['TotalCharges'], inplace=True)

cat_df = cat_df.drop('customerID', axis=1)

# Split features and target
X = cat_df.drop('Churn', axis=1)  # Target is 'Churn'
y = cat_df['Churn']

# Feature scaling (standardize numerical features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                    stratify=y, random_state=42)


# ## Model Selection & Justification  
# 
# 1. Random Forest Classifier
#     - Handling High-Dimensional Data : With 21 features, this dataset is considered high-dimensional. Random Forest can handle such datasets efficiently due to its ability to select relevant features and avoid overfitting.
#     - Robustness to Overfitting : Random Forest has an in-built mechanism to prevent overfitting through the use of ensemble methods and random feature selection. This makes it a good choice for our dataset where there's a risk of overfitting due to its size and complexity.
#     - Ability to Handle Class Imbalance : While not specifically designed for class imbalance issues, Random Forest has been shown to perform well on imbalanced datasets.
# 1. AdaBoost Classifier
#     - Adapting to Class Imbalance : AdaBoost is specifically designed to handle class imbalance problems. It can adapt its weights to focus more on the minority class during training, making it a suitable choice for our churn prediction problem.
#     - Handling High-Dimensional Data : Like Random Forest, AdaBoost can also handle high-dimensional data efficiently due to its ability to select relevant features and reduce dimensionality.
#     - Robustness to Overfitting : AdaBoost has built-in mechanisms to prevent overfitting through the use of ensemble methods and regularization.
# 1. SVM Classifier
#     - Handling Class Imbalance : SVM can be used for class imbalance problems.
#     - Robustness to Noisy Data : SVM is robust to noisy data and can handle missing values in features. This data is fairly clean, however there is likely to be a lot of noise in the data.
#     - Interpretability : SVM models are relatively interpretable due to their ability to provide feature weights, making it easier to understand which features contribute most to churn prediction.
#      
#      

# In[5]:


# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_features=None, random_state=42)
rf.fit(X_train, y_train)
# Predictions
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
# Train an SVM model
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
# Predictions for SVM
y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.decision_function(X_test)
# Train an Adaboost Classifier
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
# Predictions for Adaboost
y_pred_ada = ada.predict(X_test)
y_prob_ada = ada.predict_proba(X_test)[:, 1]
# Metrics
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()
# Metrics for SVM
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()
# Metrics for Adaboost
print("\nAdaboost Accuracy:", accuracy_score(y_test, y_pred_ada))
print("\nClassification Report:\n", classification_report(y_test, y_pred_ada))
# Confusion Matrix for Adaboost
cm_ada = confusion_matrix(y_test, y_pred_ada)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Adaboost Confusion Matrix')
plt.show()
# Encode labels
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
# ROC Curve for Random Forest and SVM
fpr_rf, tpr_rf, _= roc_curve(y_test_encoded, y_prob_rf)
auc_rf = roc_auc_score(y_test_encoded, y_prob_rf)
plt.figure(figsize=(8, 5))
plt.plot(fpr_rf, tpr_rf, label=f'RF ROC Curve (AUC = {auc_rf:.2f})')
fpr_svm, tpr_svm, _= roc_curve(y_test_encoded, y_prob_svm)
auc_svm = roc_auc_score(y_test_encoded, y_prob_svm)
plt.plot(fpr_svm, tpr_svm, label=f'SVM ROC Curve (AUC = {auc_svm:.2f})')
# Plot Adaboost ROC curve
fpr_ada, tpr_ada, _= roc_curve(y_test_encoded, y_prob_ada)
auc_ada = roc_auc_score(y_test_encoded, y_prob_ada)
plt.plot(fpr_ada, tpr_ada, label=f'Ada ROC Curve (AUC = {auc_ada:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF, SVM, and Ada ROC Curves')
plt.legend()
plt.show()


# ## Hyperparameter Tuning
# Before hyperparameter tuning for Random Forest, SVM, and Adaboost, it appears that: 
# 
#  - All three models have similar accuracy and macro avg f1-score.
#  - The precision and recall scores are relatively consistent across classes for all three models.
#  - Adaboost seems to have a slight edge in terms of overall performance, with the highest accuracy and weighted avg f1-score.
#      
# 
# Now we will run gridsearchCV to tune the parameters.

# ### For Random Forest: 
# 
# 1. The number of estimators (n_estimators) is varied between 10 and 200, which covers a broad range of possibilities from a simple model with a few decision trees to an ensemble with many decision trees. This range allows the grid search to explore both low-complexity and high-complexity models.
# 2. max_features is set to 'log2' and 'sqrt', which are common heuristics for feature selection in Random Forest, indicating that the model should consider either half or the square root of the features at each split.
# 3. min_samples_split is varied between 2 and 10, allowing the model to balance between overfitting (splitting on too few samples) and underfitting (not splitting enough).
# 4. min_samples_leaf is set to 1, 2, and 4. A lower value means the model will split more frequently, potentially improving fit but risking overfitting.
#      

# In[6]:


# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['log2', 'sqrt'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Score for Random Forest:", grid_search_rf.best_score_)
rf_tuned = grid_search_rf.best_estimator_
y_pred_rf_tuned = rf_tuned.predict(X_test)
y_prob_rf_tuned = rf_tuned.predict_proba(X_test)[:, 1]


# ### For SVM: 
# 
# 1. The regularization parameter C is varied between 0.01 and 10, which is a common range for SVM models, indicating that both low-regularization (high C) and high-regularization (low C) values are explored.
# 2. kernel is set to 'linear' and 'rbf', which are two popular kernels used in SVMs. The linear kernel does not require any transformations of the data, while the rbf (radial basis function) kernel involves a Gaussian transformation.
# 3.  gamma is varied between 'scale' and 'auto'. 'scale' uses a heuristically determined value for gamma based on the standard deviation of each feature.

# In[7]:


# Hyperparameter Tuning for SVM
param_grid_svm = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}
grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)
print("Best Parameters for SVM:", grid_search_svm.best_params_)
print("Best Score for SVM:", grid_search_svm.best_score_)
svm_tuned = grid_search_svm.best_estimator_
y_pred_svm_tuned = svm_tuned.predict(X_test)
y_prob_svm_tuned = svm_tuned.decision_function(X_test)


# ### For AdaBoost: 
# 
# 1. The number of estimators (n_estimators) is varied between 10 and 200, similar to Random Forest, allowing the model to explore both simple and complex models.
# 2. learning_rate is set to 0.1, 1, and 10, which covers a range from very low to high learning rates. A lower learning rate means smaller updates during the training process, potentially leading to overfitting, while a higher learning rate can lead to underfitting if too large.

# In[8]:


# Hyperparameter Tuning for Adaboost
param_grid_ada = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.1, 1, 10]
}
grid_search_ada = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid_ada, cv=5)
grid_search_ada.fit(X_train, y_train)
print("Best Parameters for Adaboost:", grid_search_ada.best_params_)
print("Best Score for Adaboost:", grid_search_ada.best_score_)
ada_tuned = grid_search_ada.best_estimator_
y_pred_ada_tuned = ada_tuned.predict(X_test)
y_prob_ada_tuned = ada_tuned.predict_proba(X_test)[:, 1]


# In[9]:


# Metrics
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf_tuned))
# Confusion Matrix for Random Forest
cm_rf_tuned = confusion_matrix(y_test, y_pred_rf_tuned)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf_tuned, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()
# Metrics for SVM
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm_tuned))
# Confusion Matrix for SVM
cm_svm_tuned = confusion_matrix(y_test, y_pred_svm_tuned)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm_tuned, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()
# Metrics for Adaboost
print("\nAdaboost Accuracy:", accuracy_score(y_test, y_pred_ada_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_ada))
# Confusion Matrix for Adaboost
cm_ada_tuned = confusion_matrix(y_test, y_pred_ada_tuned)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_ada_tuned, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Adaboost Confusion Matrix')
plt.show()
# Encode labels
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
# ROC Curve for Random Forest and SVM
fpr_rf_tuned, tpr_rf_tuned, _= roc_curve(y_test_encoded, y_prob_rf_tuned)
auc_rf_tuned = roc_auc_score(y_test_encoded, y_prob_rf_tuned)
plt.figure(figsize=(8, 5))
plt.plot(fpr_rf_tuned, tpr_rf_tuned, label=f'RF ROC Curve (AUC = {auc_rf:.2f})')
fpr_svm_tuned, tpr_svm_tuned, _= roc_curve(y_test_encoded, y_prob_svm_tuned)
auc_svm_tuned = roc_auc_score(y_test_encoded, y_prob_svm_tuned)
plt.plot(fpr_svm_tuned, tpr_svm_tuned, label=f'SVM ROC Curve (AUC = {auc_svm_tuned:.2f})')
# Plot Adaboost ROC curve
fpr_ada_tuned, tpr_ada_tuned, _= roc_curve(y_test_encoded, y_prob_ada_tuned)
auc_ada_tuned = roc_auc_score(y_test_encoded, y_prob_ada_tuned)
plt.plot(fpr_ada_tuned, tpr_ada_tuned, label=f'Ada ROC Curve (AUC = {auc_ada_tuned:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF, SVM, and Ada ROC Curves')
plt.legend()
plt.show()


# ## Conclusion
# After hyperparameter tuning we can see that all 3 models perform very similarly. 
# 
# For the business case we chose (best identification all potential customer churn), RandomForest performed best. While it's accuracy and AUC were slightly lower, it has the lowest number of customers at risk of churning that were misclassified as not at risk.
# 
# However, the scores were close enough that cost benefit analysis should be performed to compare the cost of retention strategies against the cost of customer churn while using the models.
# 
# Next Steps:
# After Cost Benefit Analysis identifies a model for use, the model should be deployed against live data with the chosen retention strategy.
