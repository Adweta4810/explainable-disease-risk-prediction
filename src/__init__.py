"""
src — Explainable ML for Disease Risk Prediction
=================================================
Reusable modules for the CKD and Diabetes pipelines.
"""

from src.data_preprocessing import (
    preprocess_ckd,
    preprocess_diabetes,
    load_cleaned_ckd,
    load_cleaned_diabetes,
)

from src.model_training import (
    train_all_models,
    load_models,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)

from src.evaluation import (
    compute_metrics,
    evaluate_all_models,
    overfitting_check,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_calibration_curves,
    plot_feature_importances,
    save_results_csv,
)

from src.shap_explainer import SHAPExplainer
from src.lime_explainer import LIMEExplainer

__all__ = [
    # data_preprocessing
    "preprocess_ckd", "preprocess_diabetes",
    "load_cleaned_ckd", "load_cleaned_diabetes",
    # model_training
    "train_all_models", "load_models",
    "build_logistic_regression", "build_random_forest", "build_xgboost",
    # evaluation
    "compute_metrics", "evaluate_all_models", "overfitting_check",
    "plot_confusion_matrices", "plot_roc_curves", "plot_calibration_curves",
    "plot_feature_importances", "save_results_csv",
    # explainability
    "SHAPExplainer", "LIMEExplainer",
]
