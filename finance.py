# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r'C:/Users/Ganesh Babu/Desktop/Final Project/Project 4 - Finance Loan Default Prediction/loan_default_prediction_project.csv')

# Data Preprocessing

# Handle missing values
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Impute missing values (mean for numerical, most frequent for categorical)
imputer_num = SimpleImputer(strategy='mean')
data[numerical_features] = imputer_num.fit_transform(data[numerical_features])

imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_features] = imputer_cat.fit_transform(data[categorical_features])

# Encode categorical variables
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature scaling (Standardization)
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Splitting the data into training and testing sets
# Assuming 'Loan_Status' is the target variable (replace 'loan_default' with 'Loan_Status')
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Building
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Model Evaluation

# Predict on the test set
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC/AUC

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC Score & Curve

# Compute AUC-ROC
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')  # Save the ROC curve
plt.close()

# Feature Importance (Interpretability)

# Feature importance from Random Forest model
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.savefig('feature_importance.png')  # Save the feature importance plot
plt.close()
