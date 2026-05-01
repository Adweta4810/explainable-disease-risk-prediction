# Explainable Machine Learning Approach for Disease Risk Prediction

This project develops an explainable machine learning prototype for predicting the risk of Chronic Kidney Disease (CKD) and Diabetes using patient health record data. The system trains and evaluates machine learning models, selects the best-performing models, and explains predictions using SHAP and LIME.

> This project is for academic and prototype purposes only. It does not provide medical diagnosis, treatment advice, or real-world clinical decision-making.

## Project Aim

The aim of this project is to build an explainable machine learning system that predicts CKD and Diabetes risk and provides clear explanations for each prediction.

## Main Features

- Load CKD and Diabetes datasets
- Clean and preprocess health data
- Handle missing values and incorrect data types
- Encode categorical variables
- Train and test machine learning models
- Evaluate models using Accuracy, F1-score, ROC-AUC, Precision, Recall, and Confusion Matrix
- Generate SHAP explanations
- Generate LIME explanations
- Save charts and outputs outside the notebooks
- Provide clear risk prediction results

## Machine Learning Models

The project experiments with models such as:

- Logistic Regression
- Random Forest
- XGBoost

## Explainability Methods

### SHAP

SHAP is used to explain how each feature contributes to the final prediction. It helps identify important risk factors for CKD and Diabetes.

### LIME

LIME is used to explain individual patient predictions by showing which features influenced the model output.
