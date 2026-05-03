"""
test_imports.py
───────────────
Quick smoke test — verifies all src modules import correctly and
every public function / class is accessible.

No data, no training, no plots. Runs in seconds.

Usage
-----
    python test_imports.py
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

PASS = "  ✓"
FAIL = "  ✗"
results = []


def check(label, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        results.append((label, True, None))
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"       → {e}")
        results.append((label, False, e))


print("\n" + "=" * 56)
print("  SRC MODULE SMOKE TEST")
print("=" * 56)

# ── data_preprocessing ────────────────────────────────────
print("\n[ data_preprocessing ]")

import src.data_preprocessing as dp
check("import module", lambda: None)
check("preprocess_ckd callable",        lambda: callable(dp.preprocess_ckd))
check("preprocess_diabetes callable",   lambda: callable(dp.preprocess_diabetes))
check("load_cleaned_ckd callable",      lambda: callable(dp.load_cleaned_ckd))
check("load_cleaned_diabetes callable", lambda: callable(dp.load_cleaned_diabetes))

# ── model_training ────────────────────────────────────────
print("\n[ model_training ]")

import src.model_training as mt
check("import module", lambda: None)
check("train_all_models callable",         lambda: callable(mt.train_all_models))
check("load_models callable",              lambda: callable(mt.load_models))
check("build_logistic_regression callable",lambda: callable(mt.build_logistic_regression))
check("build_random_forest callable",      lambda: callable(mt.build_random_forest))
check("build_xgboost callable",            lambda: callable(mt.build_xgboost))

# ── evaluation ────────────────────────────────────────────
print("\n[ evaluation ]")

import src.evaluation as ev
check("import module", lambda: None)
check("compute_metrics callable",         lambda: callable(ev.compute_metrics))
check("evaluate_all_models callable",     lambda: callable(ev.evaluate_all_models))
check("overfitting_check callable",       lambda: callable(ev.overfitting_check))
check("plot_confusion_matrices callable", lambda: callable(ev.plot_confusion_matrices))
check("plot_roc_curves callable",         lambda: callable(ev.plot_roc_curves))
check("plot_calibration_curves callable", lambda: callable(ev.plot_calibration_curves))
check("plot_feature_importances callable",lambda: callable(ev.plot_feature_importances))
check("save_results_csv callable",        lambda: callable(ev.save_results_csv))

# ── shap_explainer ────────────────────────────────────────
print("\n[ shap_explainer ]")

import src.shap_explainer as se
check("import module", lambda: None)
check("SHAPExplainer is a class", lambda: isinstance(se.SHAPExplainer, type))
for method in [
    "compute_shap_values", "plot_summary", "plot_feature_importance_bar",
    "plot_force", "plot_waterfall", "plot_dependence",
    "get_importance_table", "run_full_explanation",
]:
    check(f"SHAPExplainer.{method} exists",
          lambda m=method: callable(getattr(se.SHAPExplainer, m)))

# ── lime_explainer ────────────────────────────────────────
print("\n[ lime_explainer ]")

import src.lime_explainer as le
check("import module", lambda: None)
check("LIMEExplainer is a class", lambda: isinstance(le.LIMEExplainer, type))
for method in [
    "explain_patient", "explain_multiple_patients",
    "run_stability_check", "global_feature_importance",
]:
    check(f"LIMEExplainer.{method} exists",
          lambda m=method: callable(getattr(le.LIMEExplainer, m)))

# ── third-party dependencies ──────────────────────────────
print("\n[ third-party dependencies ]")

for pkg, import_name in [
    ("pandas",           "pandas"),
    ("numpy",            "numpy"),
    ("scikit-learn",     "sklearn"),
    ("imbalanced-learn", "imblearn"),
    ("xgboost",          "xgboost"),
    ("shap",             "shap"),
    ("lime",             "lime"),
    ("matplotlib",       "matplotlib"),
    ("seaborn",          "seaborn"),
    ("joblib",           "joblib"),
]:
    check(f"{pkg}", lambda m=import_name: __import__(m))

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 56)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print(f"  {passed}/{total} checks passed", end="")
if failed == 0:
    print("  — all good! ✓")
else:
    print(f"  — {failed} failed ✗")
    print("\n  Failed checks:")
    for label, ok, err in results:
        if not ok:
            print(f"    • {label}: {err}")
print("=" * 56 + "\n")

sys.exit(0 if failed == 0 else 1)