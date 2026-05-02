"""
data_preprocessing.py
─────────────────────
Reusable preprocessing pipeline for CKD and Diabetes datasets.

Usage
-----
from src.data_preprocessing import preprocess_ckd, preprocess_diabetes

ckd_data      = preprocess_ckd("data/chronic_kidney_disease/kidney_disease.csv")
diabetes_data = preprocess_diabetes("data/diabetes/diabetes.csv")

Each function returns a dict with keys:
    X_train, X_test, y_train, y_test,
    X_train_scaled, X_test_scaled,
    X_train_resampled, y_train_resampled,
    scaler, feature_names, cleaned_df
"""

import os
import warnings
import collections

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# ──────────────────────────────────────────────
# CKD
# ──────────────────────────────────────────────

def preprocess_ckd(
    raw_path: str,
    save_cleaned_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Full preprocessing pipeline for the Chronic Kidney Disease dataset.

    Parameters
    ----------
    raw_path : str
        Path to the raw kidney_disease.csv file.
    save_cleaned_path : str, optional
        If provided, saves the cleaned DataFrame as a CSV at this path.
    test_size : float
        Fraction of data held out for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with all splits, scaler, feature names, and cleaned_df.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(raw_path)

    # ── 2. Replace dirty missing markers ────────────────────────────────────
    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)

    # ── 3. Strip whitespace from object columns ──────────────────────────────
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    # ── 4. Drop ID column ────────────────────────────────────────────────────
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # ── 5. Cast numeric-like columns ─────────────────────────────────────────
    numeric_like = [
        "age", "bp", "sg", "al", "su", "bgr", "bu",
        "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 6. Standardise text values ───────────────────────────────────────────
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.lower().str.strip()

    df = df.replace(
        {
            "\tyes": "yes", "\tno": "no",
            " yes": "yes",  " no": "no",
            "ckd\t": "ckd", "not ckd": "notckd",
        }
    )

    # ── 7. Encode categorical columns ────────────────────────────────────────
    mapping_dict = {
        "rbc":            {"normal": 0, "abnormal": 1},
        "pc":             {"normal": 0, "abnormal": 1},
        "pcc":            {"notpresent": 0, "present": 1},
        "ba":             {"notpresent": 0, "present": 1},
        "htn":            {"no": 0, "yes": 1},
        "dm":             {"no": 0, "yes": 1},
        "cad":            {"no": 0, "yes": 1},
        "appet":          {"good": 0, "poor": 1},
        "pe":             {"no": 0, "yes": 1},
        "ane":            {"no": 0, "yes": 1},
        "classification": {"notckd": 0, "ckd": 1},
    }
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # ── 8. Convert mapped columns to numeric ─────────────────────────────────
    mapped_cols = list(mapping_dict.keys())
    for col in mapped_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 9. Fill missing values ───────────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── 10. Outlier clipping (IQR) on continuous columns ────────────────────
    binary_cols = [
        "rbc", "pc", "pcc", "ba", "htn", "dm",
        "cad", "appet", "pe", "ane", "classification",
    ]
    continuous_cols = [c for c in df.columns if c not in binary_cols]
    for col in continuous_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # ── 11. Save cleaned CSV ─────────────────────────────────────────────────
    if save_cleaned_path:
        os.makedirs(os.path.dirname(save_cleaned_path), exist_ok=True)
        df.to_csv(save_cleaned_path, index=False)
        print(f"[CKD] Cleaned dataset saved → {save_cleaned_path}")

    # ── 12. Split features / target ──────────────────────────────────────────
    X = df.drop(columns=["classification"])
    y = df["classification"]

    # ── 13. Train / test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 14. Feature scaling ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns
    )

    # ── 15. SMOTE (train only) ───────────────────────────────────────────────
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )

    print("[CKD] Preprocessing complete.")
    print(f"  Train (after SMOTE) : {X_train_resampled.shape}")
    print(f"  Test                : {X_test_scaled.shape}")
    print(f"  Class balance       : {collections.Counter(y_train_resampled)}")

    return {
        "X_train":           X_train,
        "X_test":            X_test,
        "y_train":           y_train,
        "y_test":            y_test,
        "X_train_scaled":    X_train_scaled,
        "X_test_scaled":     X_test_scaled,
        "X_train_resampled": X_train_resampled,
        "y_train_resampled": y_train_resampled,
        "scaler":            scaler,
        "feature_names":     list(X.columns),
        "cleaned_df":        df,
    }


# ──────────────────────────────────────────────
# Diabetes
# ──────────────────────────────────────────────

def preprocess_diabetes(
    raw_path: str,
    save_cleaned_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Full preprocessing pipeline for the Pima Indians Diabetes dataset.

    Parameters
    ----------
    raw_path : str
        Path to the raw diabetes.csv file.
    save_cleaned_path : str, optional
        If provided, saves the cleaned DataFrame as a CSV at this path.
    test_size : float
        Fraction of data held out for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with all splits, scaler, feature names, and cleaned_df.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(raw_path)

    # ── 2. Replace biologically invalid zeros with NaN ───────────────────────
    invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in invalid_zero_cols:
        df[col] = df[col].replace(0, np.nan)

    # ── 3. Remove duplicates ─────────────────────────────────────────────────
    df = df.drop_duplicates()

    # ── 4. Outlier clipping (IQR) ────────────────────────────────────────────
    cols_to_clip = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_to_clip:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # ── 5. Split features / target ───────────────────────────────────────────
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # ── 6. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 7. Impute missing values (median, fit on train only) ─────────────────
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train[numerical_cols]),
        columns=numerical_cols,
        index=X_train.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test[numerical_cols]),
        columns=numerical_cols,
        index=X_test.index,
    )
    # Re-attach any non-numeric columns (none expected, but future-safe)
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[numerical_cols] = X_train_imp
    X_test[numerical_cols]  = X_test_imp

    # ── 8. Feature scaling (fit on train only) ───────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # ── 9. SMOTE (train only) ────────────────────────────────────────────────
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )

    # ── 10. Save cleaned CSV (pre-split, pre-scale snapshot) ─────────────────
    if save_cleaned_path:
        cleaned_snapshot = pd.concat([X, y], axis=1)
        os.makedirs(os.path.dirname(save_cleaned_path), exist_ok=True)
        cleaned_snapshot.to_csv(save_cleaned_path, index=False)
        print(f"[Diabetes] Cleaned dataset saved → {save_cleaned_path}")

    print("[Diabetes] Preprocessing complete.")
    print(f"  Train (after SMOTE) : {X_train_resampled.shape}")
    print(f"  Test                : {X_test_scaled.shape}")
    print(f"  Class balance       : {collections.Counter(y_train_resampled)}")

    return {
        "X_train":           X_train,
        "X_test":            X_test,
        "y_train":           y_train,
        "y_test":            y_test,
        "X_train_scaled":    X_train_scaled,
        "X_test_scaled":     X_test_scaled,
        "X_train_resampled": X_train_resampled,
        "y_train_resampled": y_train_resampled,
        "scaler":            scaler,
        "feature_names":     list(X.columns),
        "cleaned_df":        df,
    }


# ──────────────────────────────────────────────
# Convenience: load from already-cleaned CSV
# ──────────────────────────────────────────────

def load_cleaned_ckd(
    cleaned_path: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Load a pre-cleaned CKD CSV (output of preprocess_ckd) and re-split/scale.
    Useful when the raw cleaning has already been done once.
    """
    df = pd.read_csv(cleaned_path)
    X  = df.drop(columns=["classification"])
    y  = df["classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "X_train_resampled": X_train_resampled, "y_train_resampled": y_train_resampled,
        "scaler": scaler, "feature_names": list(X.columns), "cleaned_df": df,
    }


def load_cleaned_diabetes(
    cleaned_path: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Load a pre-cleaned Diabetes CSV (output of preprocess_diabetes) and re-split/scale.
    """
    df = pd.read_csv(cleaned_path)
    X  = df.drop(columns=["Outcome"])
    y  = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "X_train_resampled": X_train_resampled, "y_train_resampled": y_train_resampled,
        "scaler": scaler, "feature_names": list(X.columns), "cleaned_df": df,
    }
