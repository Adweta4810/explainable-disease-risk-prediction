"""
evaluation.py
─────────────
Reusable model-evaluation utilities.

Usage
-----
from src.evaluation import evaluate_all_models, plot_confusion_matrices,
                           plot_roc_curves, plot_calibration_curves,
                           plot_feature_importances, overfitting_check

# After training:
test_results = evaluate_all_models(models_dict, X_test, y_test, dataset_label="CKD")
plot_confusion_matrices(models_dict, X_test, y_test, output_dir="outputs/CKD/plots")
plot_roc_curves(models_dict, X_test, y_test, output_dir="outputs/CKD/plots")
plot_calibration_curves(models_dict, X_test, y_test, output_dir="outputs/CKD/plots")
plot_feature_importances(models_dict, feature_names, output_dir="outputs/CKD/plots")
overfitting_check(models_dict, cv_results, test_results)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    cohen_kappa_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute the standard set of classification metrics."""
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
        "F1":        f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "Kappa":     cohen_kappa_score(y_true, y_pred),
        "Brier":     brier_score_loss(y_true, y_prob),
    }


def evaluate_all_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_label: str = "CKD",
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Evaluate all models on the hold-out test set.

    Parameters
    ----------
    models : dict
        Mapping {model_name: fitted_pipeline}.
    X_test, y_test : hold-out test data (RAW — each pipeline handles scaling).
    dataset_label : printed in the summary header.
    class_names : e.g. ["Not CKD", "CKD"] or ["No Diabetes", "Diabetes"].

    Returns
    -------
    pd.DataFrame with one row per model and metric columns.
    """
    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    results = {}

    print(f"\n{'='*60}")
    print(f"  Hold-out Test Results — {dataset_label}")
    print(f"{'='*60}")

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        results[name] = metrics

        print(f"\n── {name} ──")
        for k, v in metrics.items():
            direction = " ← lower = better" if k == "Brier" else ""
            print(f"  {k:<12}: {v:.4f}{direction}")
        print(classification_report(y_test, y_pred, target_names=class_names))

    df = pd.DataFrame(results).T
    df.index.name = "Model"

    print(f"\n{'='*60}")
    print("  Summary Table")
    print(f"{'='*60}")
    print(df.round(4).to_string())
    print(f"\n  Best by ROC-AUC : {df['ROC-AUC'].idxmax()}")
    print(f"  Best by Kappa   : {df['Kappa'].idxmax()}")
    print(f"  Best by F1      : {df['F1'].idxmax()}")

    return df


# ──────────────────────────────────────────────
# Overfitting check
# ──────────────────────────────────────────────

def overfitting_check(
    models: dict,
    cv_results: dict,
    test_results: pd.DataFrame,
    threshold: float = 0.05,
) -> None:
    """
    Print CV vs hold-out AUC gap for each model.

    Parameters
    ----------
    models : dict mapping model_name → pipeline (used for ordering only).
    cv_results : dict mapping model_name → dict with "ROC-AUC" key.
    test_results : DataFrame returned by evaluate_all_models.
    threshold : gap above which a model is flagged as overfitting (default 0.05).
    """
    print(f"\n{'='*60}")
    print("  Overfitting Check — CV vs Hold-out Gap")
    print(f"{'='*60}")
    print(f"{'Model':<22} {'CV AUC':>8}  {'Test AUC':>9}  {'Gap':>8}  Status")
    print("-" * 58)

    for name in models:
        cv_auc   = cv_results[name]["ROC-AUC"] if isinstance(cv_results[name], dict) else cv_results[name]
        test_auc = test_results.loc[name, "ROC-AUC"]
        gap      = cv_auc - test_auc
        status   = "⚠ Overfit" if gap > threshold else "✓ OK"
        print(f"{name:<22} {cv_auc:>8.4f}  {test_auc:>9.4f}  {gap:>+8.4f}  {status}")


# ──────────────────────────────────────────────
# Plot utilities
# ──────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def plot_confusion_matrices(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str] | None = None,
    output_dir: str | None = None,
    filename: str = "confusion_matrices.png",
) -> None:
    """Plot side-by-side confusion matrices for all models."""
    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    _ensure_dir(output_dir)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_roc_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | None = None,
    filename: str = "roc_curves.png",
    title: str = "ROC Curves — Hold-out Test Set",
) -> None:
    """Plot ROC curves for all models on the same axes."""
    _ensure_dir(output_dir)

    line_styles = ["-", "--", ":"]
    markers     = ["o", "s", "^"]

    plt.figure(figsize=(8, 6))

    for i, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(
            fpr, tpr,
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            linewidth=2, markersize=6,
            label=f"{name} (AUC = {auc:.4f})",
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_dir:
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_calibration_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 10,
    output_dir: str | None = None,
    filename: str = "calibration_curves.png",
) -> None:
    """Plot calibration curves for all models."""
    _ensure_dir(output_dir)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, fraction_pos, "s-", label=name)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_title(f"Calibration — {name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_feature_importances(
    models: dict,
    feature_names: list[str],
    top_n: int = 15,
    output_dir: str | None = None,
    filename: str = "feature_importances.png",
) -> None:
    """
    Plot feature importances for tree-based models (Random Forest, XGBoost).
    Logistic Regression is skipped automatically.
    """
    _ensure_dir(output_dir)

    # Filter to models with feature_importances_
    tree_models = {
        name: model for name, model in models.items()
        if hasattr(model.named_steps.get("classifier"), "feature_importances_")
    }
    if not tree_models:
        print("No tree-based models found — skipping feature importance plot.")
        return

    fig, axes = plt.subplots(1, len(tree_models), figsize=(8 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        clf = model.named_steps["classifier"]
        importances = pd.Series(clf.feature_importances_, index=feature_names)
        importances.nlargest(top_n).sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Top-{top_n} Feature Importances\n{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def save_results_csv(
    test_results: pd.DataFrame,
    output_dir: str,
    filename: str = "model_results.csv",
) -> None:
    """Save the test-results DataFrame to CSV."""
    _ensure_dir(output_dir)
    save_path = os.path.join(output_dir, filename)
    test_results.round(4).to_csv(save_path)
    print(f"Results saved → {save_path}")
