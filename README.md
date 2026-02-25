# Explainable Disease Risk Prediction (CKD + Diabetes)

An end-to-end **Explainable Machine Learning (XAI)** project that predicts the risk of **Chronic Kidney Disease (CKD)** and **Diabetes** using public datasets. The system trains and evaluates classification models (ROC-AUC, F1-score) and explains predictions using **SHAP** (primary) and **LIME** (secondary). A simple dashboard will be included for user-friendly testing of predictions and explanations.

> **Disclaimer:** This project is for academic/educational use only and must not be used for real medical diagnosis or clinical decision-making.

---

## What this repository contains
This repository contains:
- Raw datasets (small public files) and dataset provenance notes
- Notebooks/scripts for data preprocessing, model training, and evaluation
- Explainability outputs (SHAP/LIME) for global and local interpretation
- A lightweight dashboard (Streamlit) to input values and view predictions

---

## Datasets
- **CKD (UCI):** Chronic Kidney Disease dataset  
  https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

- **Diabetes (Kaggle mirror):** Pima Indians Diabetes Database  
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Current dataset notes are recorded in: **`data_source.md`**

---

## Tech stack
- **Python 3.10+**
- **VS Code** (recommended)
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, SHAP, LIME
- **Dashboard:** Streamlit
