"""
run_pipeline.py
───────────────
Full end-to-end pipeline for the Explainable ML Disease Risk Prediction project.
Runs preprocessing, model training, evaluation, SHAP, and LIME for both
CKD and Diabetes datasets.

Usage
-----
    python run_pipeline.py                        # run everything
    python run_pipeline.py --disease ckd          # CKD only
    python run_pipeline.py --disease diabetes     # Diabetes only
    python run_pipeline.py --skip-explainability  # skip SHAP + LIME (faster)

Project structure expected
--------------------------
    project/
    ├── run_pipeline.py
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── model_training.py
    │   ├── evaluation.py
    │   ├── shap_explainer.py
    │   └── lime_explainer.py
    ├── data/
    │   ├── chronic_kidney_disease/
    │   │   └── kidney_disease.csv
    │   └── diabetes/
    │       └── diabetes.csv
    ├── models/
    │   ├── CKD/
    │   └── Diabetes/
    └── outputs/
        ├── CKD/
        │   ├── plots/
        │   ├── SHAP/
        │   └── LIME/
        └── Diabetes/
            ├── plots/
            ├── SHAP/
            └── LIME/
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensures src/ is importable when run from the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.data_preprocessing import preprocess_ckd, preprocess_diabetes
from src.model_training      import train_all_models
from src.evaluation          import (
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


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

CKD_RAW_PATH          = os.path.join(ROOT_DIR, "data", "chronic_kidney_disease", "kidney_disease.csv")
CKD_CLEANED_PATH      = os.path.join(ROOT_DIR, "data", "chronic_kidney_disease", "ckd_cleaned.csv")
CKD_MODEL_DIR         = os.path.join(ROOT_DIR, "models", "CKD")
CKD_PLOT_DIR          = os.path.join(ROOT_DIR, "outputs", "CKD", "plots")
CKD_SHAP_DIR          = os.path.join(ROOT_DIR, "outputs", "CKD", "SHAP")
CKD_LIME_DIR          = os.path.join(ROOT_DIR, "outputs", "CKD", "LIME")

DIABETES_RAW_PATH     = os.path.join(ROOT_DIR, "data", "diabetes", "diabetes.csv")
DIABETES_CLEANED_PATH = os.path.join(ROOT_DIR, "data", "diabetes", "diabetes_cleaned.csv")
DIABETES_MODEL_DIR    = os.path.join(ROOT_DIR, "models", "Diabetes")
DIABETES_PLOT_DIR     = os.path.join(ROOT_DIR, "outputs", "Diabetes", "plots")
DIABETES_SHAP_DIR     = os.path.join(ROOT_DIR, "outputs", "Diabetes", "SHAP")
DIABETES_LIME_DIR     = os.path.join(ROOT_DIR, "outputs", "Diabetes", "LIME")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _section(text: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print("─" * 50)


def _elapsed(start: float) -> str:
    secs = time.time() - start
    return f"{secs // 60:.0f}m {secs % 60:.0f}s"


# ──────────────────────────────────────────────────────────────────────────────
# CKD pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_ckd(skip_explainability: bool = False) -> None:
    t0 = time.time()
    _banner("CKD PIPELINE — Chronic Kidney Disease")

    # ── 1. Preprocessing ─────────────────────────────────────────────────────
    _section("Step 1 — Data Preprocessing")
    ckd = preprocess_ckd(
        raw_path=CKD_RAW_PATH,
        save_cleaned_path=CKD_CLEANED_PATH,
    )
    X_train      = ckd["X_train"]
    X_test       = ckd["X_test"]
    y_train      = ckd["y_train"]
    y_test       = ckd["y_test"]
    feature_names = ckd["feature_names"]

    # ── 2. Model Training ────────────────────────────────────────────────────
    _section("Step 2 — Model Training (LR / RF / XGBoost)")
    training_results = train_all_models(
        X_train=X_train,
        y_train=y_train,
        dataset_label="CKD",
        model_save_dir=CKD_MODEL_DIR,
        xgb_include_scaler=False,   # CKD: pipeline handles scaling via LR; RF/XGB use raw
    )
    models     = {name: r["model"]      for name, r in training_results.items()}
    cv_results = {name: r["cv_metrics"] for name, r in training_results.items()}

    # ── 3. Evaluation ────────────────────────────────────────────────────────
    _section("Step 3 — Evaluation")
    test_results = evaluate_all_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        dataset_label="CKD",
        class_names=["Not CKD", "CKD"],
    )

    overfitting_check(models, cv_results, test_results)

    os.makedirs(CKD_PLOT_DIR, exist_ok=True)

    plot_confusion_matrices(
        models, X_test, y_test,
        class_names=["Not CKD", "CKD"],
        output_dir=CKD_PLOT_DIR,
    )
    plot_roc_curves(
        models, X_test, y_test,
        output_dir=CKD_PLOT_DIR,
        title="ROC Curves — CKD Hold-out Test Set",
    )
    plot_calibration_curves(
        models, X_test, y_test,
        output_dir=CKD_PLOT_DIR,
    )
    plot_feature_importances(
        models, feature_names,
        output_dir=CKD_PLOT_DIR,
    )
    save_results_csv(
        test_results,
        output_dir=CKD_PLOT_DIR,
        filename="ckd_model_results.csv",
    )

    # ── 4. SHAP Explainability ───────────────────────────────────────────────
    if not skip_explainability:
        _section("Step 4 — SHAP Explainability (Random Forest)")

        rf_model = models["Random Forest"]
        y_pred_rf = rf_model.predict(X_test)

        shap_exp = SHAPExplainer(
            model_pipeline=rf_model,
            X_test=X_test,
            feature_names=feature_names,
            class_names=["Not CKD", "CKD"],
            output_dir=CKD_SHAP_DIR,
            dataset_label="CKD",
            scale_for_shap=False,   # RF pipeline has no scaler step
        )
        shap_exp.run_full_explanation(
            patient_idx=0,
            y_test=y_test,
            y_pred=y_pred_rf,
        )

        # ── 5. LIME Explainability ────────────────────────────────────────────
        _section("Step 5 — LIME Explainability (Random Forest)")

        lime_exp = LIMEExplainer(
            model_pipeline=rf_model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            class_names=["Not CKD", "CKD"],
            output_dir=CKD_LIME_DIR,
            dataset_label="CKD",
        )
        lime_exp.explain_patient(patient_idx=0, y_test=y_test)
        lime_exp.run_stability_check(patient_idx=0, n_runs=5)

    _banner(f"CKD PIPELINE COMPLETE  ({_elapsed(t0)})")


# ──────────────────────────────────────────────────────────────────────────────
# Diabetes pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_diabetes(skip_explainability: bool = False) -> None:
    t0 = time.time()
    _banner("DIABETES PIPELINE — Pima Indians Diabetes")

    # ── 1. Preprocessing ─────────────────────────────────────────────────────
    _section("Step 1 — Data Preprocessing")
    dia = preprocess_diabetes(
        raw_path=DIABETES_RAW_PATH,
        save_cleaned_path=DIABETES_CLEANED_PATH,
    )
    X_train       = dia["X_train"]
    X_test        = dia["X_test"]
    y_train       = dia["y_train"]
    y_test        = dia["y_test"]
    feature_names = dia["feature_names"]

    # ── 2. Model Training ────────────────────────────────────────────────────
    _section("Step 2 — Model Training (LR / RF / XGBoost)")
    training_results = train_all_models(
        X_train=X_train,
        y_train=y_train,
        dataset_label="Diabetes",
        model_save_dir=DIABETES_MODEL_DIR,
        xgb_include_scaler=True,   # Diabetes XGBoost pipeline includes its own scaler
    )
    models     = {name: r["model"]      for name, r in training_results.items()}
    cv_results = {name: r["cv_metrics"] for name, r in training_results.items()}

    # ── 3. Evaluation ────────────────────────────────────────────────────────
    _section("Step 3 — Evaluation")
    test_results = evaluate_all_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        dataset_label="Diabetes",
        class_names=["No Diabetes", "Diabetes"],
    )

    overfitting_check(models, cv_results, test_results)

    os.makedirs(DIABETES_PLOT_DIR, exist_ok=True)

    plot_confusion_matrices(
        models, X_test, y_test,
        class_names=["No Diabetes", "Diabetes"],
        output_dir=DIABETES_PLOT_DIR,
    )
    plot_roc_curves(
        models, X_test, y_test,
        output_dir=DIABETES_PLOT_DIR,
        title="ROC Curves — Diabetes Hold-out Test Set",
    )
    plot_calibration_curves(
        models, X_test, y_test,
        output_dir=DIABETES_PLOT_DIR,
    )
    plot_feature_importances(
        models, feature_names,
        output_dir=DIABETES_PLOT_DIR,
    )
    save_results_csv(
        test_results,
        output_dir=DIABETES_PLOT_DIR,
        filename="diabetes_model_results.csv",
    )

    # ── 4. SHAP Explainability ───────────────────────────────────────────────
    if not skip_explainability:
        _section("Step 4 — SHAP Explainability (XGBoost)")

        xgb_model = models["XGBoost"]
        y_pred_xgb = xgb_model.predict(X_test)

        shap_exp = SHAPExplainer(
            model_pipeline=xgb_model,
            X_test=X_test,
            feature_names=feature_names,
            class_names=["No Diabetes", "Diabetes"],
            output_dir=DIABETES_SHAP_DIR,
            dataset_label="Diabetes",
            scale_for_shap=True,   # extract scaler from XGBoost pipeline
        )
        shap_exp.run_full_explanation(
            patient_idx=0,
            y_test=y_test,
            y_pred=y_pred_xgb,
        )

        # ── 5. LIME Explainability ────────────────────────────────────────────
        _section("Step 5 — LIME Explainability (XGBoost)")

        lime_exp = LIMEExplainer(
            model_pipeline=xgb_model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            class_names=["No Diabetes", "Diabetes"],
            output_dir=DIABETES_LIME_DIR,
            dataset_label="Diabetes",
        )
        lime_exp.explain_patient(patient_idx=0, y_test=y_test)
        lime_exp.run_stability_check(patient_idx=0, n_runs=5)

    _banner(f"DIABETES PIPELINE COMPLETE  ({_elapsed(t0)})")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explainable ML pipeline for CKD and Diabetes risk prediction."
    )
    parser.add_argument(
        "--disease",
        choices=["ckd", "diabetes", "both"],
        default="both",
        help="Which disease pipeline to run (default: both).",
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip SHAP and LIME steps (useful for quick evaluation runs).",
    )
    args = parser.parse_args()

    total_start = time.time()

    _banner("EXPLAINABLE ML — DISEASE RISK PREDICTION")
    print(f"  Disease   : {args.disease}")
    print(f"  SHAP/LIME : {'SKIPPED' if args.skip_explainability else 'enabled'}")

    # Validate data files exist before starting
    paths_to_check = []
    if args.disease in ("ckd", "both"):
        paths_to_check.append(("CKD raw data", CKD_RAW_PATH))
    if args.disease in ("diabetes", "both"):
        paths_to_check.append(("Diabetes raw data", DIABETES_RAW_PATH))

    missing = [(label, p) for label, p in paths_to_check if not os.path.exists(p)]
    if missing:
        print("\n[ERROR] Missing data files:")
        for label, path in missing:
            print(f"  ✗  {label}: {path}")
        print("\nPlease ensure your data files are in the expected locations.")
        sys.exit(1)

    print("\n  Data files found ✓")

    # Run pipeline(s)
    if args.disease in ("ckd", "both"):
        run_ckd(skip_explainability=args.skip_explainability)

    if args.disease in ("diabetes", "both"):
        run_diabetes(skip_explainability=args.skip_explainability)

    _banner(f"ALL PIPELINES COMPLETE  (total: {_elapsed(total_start)})")
    print("\n  Output locations:")
    print(f"  Models  → models/")
    print(f"  Plots   → outputs/CKD/plots/  &  outputs/Diabetes/plots/")
    print(f"  SHAP    → outputs/CKD/SHAP/   &  outputs/Diabetes/SHAP/")
    print(f"  LIME    → outputs/CKD/LIME/   &  outputs/Diabetes/LIME/")
    print()


if __name__ == "__main__":
    main()