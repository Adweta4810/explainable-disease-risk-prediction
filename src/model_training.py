"""
model_training.py
─────────────────
Reusable model-training utilities for CKD and Diabetes pipelines.

Each build_* function returns a fitted best estimator (ImbPipeline).
train_all_models() trains all three in one call and returns a results dict.

Usage
-----
from src.model_training import train_all_models

results = train_all_models(
    X_train, y_train,
    dataset_label="CKD",          # "CKD" or "Diabetes"
    model_save_dir="models/CKD",  # set None to skip saving
)

# results keys: "Logistic Regression", "Random Forest", "XGBoost"
# Each value: {"model": fitted_pipeline, "cv_auc": float, "best_params": dict}
"""

import os
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# ──────────────────────────────────────────────
# CV helper
# ──────────────────────────────────────────────

def _cv_report(pipeline, X_train, y_train, label: str, random_state: int = RANDOM_STATE) -> dict:
    """
    Run honest 5-fold CV (SMOTE stays inside each fold via ImbPipeline).
    Returns dict with Accuracy, ROC-AUC, Kappa means.
    """
    skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    kappa_scorer = make_scorer(cohen_kappa_score)

    acc   = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
    roc   = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc")
    kappa = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring=kappa_scorer)

    print(f"\n[{label}]  5-Fold CV")
    print(f"  Accuracy : {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  ROC-AUC  : {roc.mean():.4f} ± {roc.std():.4f}")
    print(f"  Kappa    : {kappa.mean():.4f} ± {kappa.std():.4f}")

    return {"Accuracy": acc.mean(), "ROC-AUC": roc.mean(), "Kappa": kappa.mean()}


# ──────────────────────────────────────────────
# Individual model builders
# ──────────────────────────────────────────────

def build_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Train a tuned Logistic Regression inside an ImbPipeline (scaler + SMOTE).

    Returns
    -------
    dict: model, cv_auc, best_params
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    pipeline = ImbPipeline([
        ("scaler",     StandardScaler()),
        ("smote",      SMOTE(random_state=random_state)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )),
    ])

    param_grid = {
        "classifier__C":       [0.05, 0.1, 0.5],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver":  ["liblinear"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    cv_metrics = _cv_report(best_model, X_train, y_train, "Logistic Regression", random_state)

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")

    return {
        "model":       best_model,
        "cv_auc":      search.best_score_,
        "cv_metrics":  cv_metrics,
        "best_params": search.best_params_,
    }


def build_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Train a tuned Random Forest inside an ImbPipeline (SMOTE inside CV).

    Returns
    -------
    dict: model, cv_auc, best_params
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    pipeline = ImbPipeline([
        ("smote",      SMOTE(random_state=random_state)),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "classifier__n_estimators":      [100, 200],
        "classifier__max_depth":         [4, 6, 8],
        "classifier__min_samples_split": [8, 12],
        "classifier__min_samples_leaf":  [3, 5],
        "classifier__max_features":      ["sqrt"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    cv_metrics = _cv_report(best_model, X_train, y_train, "Random Forest", random_state)

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")

    return {
        "model":       best_model,
        "cv_auc":      search.best_score_,
        "cv_metrics":  cv_metrics,
        "best_params": search.best_params_,
    }


def build_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    include_scaler: bool = False,
    random_state: int = RANDOM_STATE,
    n_iter: int = 30,
) -> dict:
    """
    Train a tuned XGBoost inside an ImbPipeline (SMOTE inside CV).

    Parameters
    ----------
    include_scaler : bool
        Set True for the diabetes pipeline (XGBoost benefits from scaling there).
        False for CKD (scaler already applied upstream).
    n_iter : int
        Number of iterations for RandomizedSearchCV (default 30).

    Returns
    -------
    dict: model, cv_auc, best_params
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    steps = []
    if include_scaler:
        steps.append(("scaler", StandardScaler()))
    steps += [
        ("smote",      SMOTE(random_state=random_state)),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )),
    ]

    pipeline  = ImbPipeline(steps)

    param_grid = {
        "classifier__n_estimators":    [100, 200],
        "classifier__max_depth":       [3, 4],
        "classifier__learning_rate":   [0.03, 0.05, 0.1],
        "classifier__subsample":       [0.7, 0.8],
        "classifier__colsample_bytree":[0.6, 0.7],
        "classifier__reg_alpha":       [0.5, 1.0],
        "classifier__reg_lambda":      [1.5, 2.0],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=skf,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    cv_metrics = _cv_report(best_model, X_train, y_train, "XGBoost", random_state)

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")

    return {
        "model":       best_model,
        "cv_auc":      search.best_score_,
        "cv_metrics":  cv_metrics,
        "best_params": search.best_params_,
    }


# ──────────────────────────────────────────────
# Train all three models in one call
# ──────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dataset_label: str = "CKD",
    model_save_dir: str | None = None,
    xgb_include_scaler: bool = False,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Train Logistic Regression, Random Forest, and XGBoost.

    Parameters
    ----------
    X_train, y_train : training data (RAW, unscaled — each pipeline handles its own scaling).
    dataset_label    : "CKD" or "Diabetes" — used in print messages and saved filenames.
    model_save_dir   : directory to save .pkl files; None = skip saving.
    xgb_include_scaler : pass True for Diabetes (XGBoost pipeline includes its own scaler).

    Returns
    -------
    dict mapping model name → {model, cv_auc, cv_metrics, best_params}
    """
    print(f"\n{'='*60}")
    print(f"  Training models for {dataset_label}")
    print(f"{'='*60}")

    print("\n── Logistic Regression ──")
    lr_result = build_logistic_regression(X_train, y_train, random_state)

    print("\n── Random Forest ──")
    rf_result = build_random_forest(X_train, y_train, random_state)

    print("\n── XGBoost ──")
    xgb_result = build_xgboost(
        X_train, y_train,
        include_scaler=xgb_include_scaler,
        random_state=random_state,
    )

    results = {
        "Logistic Regression": lr_result,
        "Random Forest":       rf_result,
        "XGBoost":             xgb_result,
    }

    # ── Save models ───────────────────────────────────────────────────────────
    if model_save_dir:
        os.makedirs(model_save_dir, exist_ok=True)
        prefix = dataset_label.lower().replace(" ", "_")
        joblib.dump(lr_result["model"],  f"{model_save_dir}/{prefix}_lr_pipeline.pkl")
        joblib.dump(rf_result["model"],  f"{model_save_dir}/{prefix}_rf_pipeline.pkl")
        joblib.dump(xgb_result["model"], f"{model_save_dir}/{prefix}_xgb_pipeline.pkl")
        print(f"\n[{dataset_label}] Models saved to {model_save_dir}/")

    return results


# ──────────────────────────────────────────────
# Load saved models
# ──────────────────────────────────────────────

def load_models(model_save_dir: str, dataset_label: str = "CKD") -> dict:
    """
    Load previously saved model pipelines from disk.

    Returns
    -------
    dict: {"Logistic Regression": pipeline, "Random Forest": pipeline, "XGBoost": pipeline}
    """
    prefix = dataset_label.lower().replace(" ", "_")
    return {
        "Logistic Regression": joblib.load(f"{model_save_dir}/{prefix}_lr_pipeline.pkl"),
        "Random Forest":       joblib.load(f"{model_save_dir}/{prefix}_rf_pipeline.pkl"),
        "XGBoost":             joblib.load(f"{model_save_dir}/{prefix}_xgb_pipeline.pkl"),
    }
