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
import traceback

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


# ──────────────────────────────────────────────
print("\n" + "=" * 56)
print("  SRC MODULE SMOKE TEST")
print("=" * 56)

# ── data_preprocessing ────────────────────────
print("\n[ data_preprocessing ]")
check("import module",
      lambda: __import__("src.data_preprocessing", fromlist=["*"]))

check("preprocess_ckd callable", lambda: (
    __import__("src.data_preprocessing", fromlist=["preprocess_ckd"]),
    callable(getattr(sys.modules["src.data_preprocessing"], "preprocess_ckd"))
))
check("preprocess_diabetes callable", lambda: (
    callable(getattr(sys.modules["src.data_preprocessing"], "preprocess_diabetes"))
))
check("load_cleaned_ckd callable", lambda: (
    callable(getattr(sys.modules["src.data_preprocessing"], "load_cleaned_ckd"))
))
check("load_cleaned_diabetes callable", lambda: (
    callable(getattr(sys.modules["src.data_preprocessing"], "load_cleaned_diabetes"))
))

# ── model_training ────────────────────────────
print("\n[ model_training ]")
check("import module",
      lambda: __import__("src.model_training", fromlist=["*"]))

for fn_name in [
    "train_all_models", "load_models",
    "build_logistic_regression", "build_random_forest", "build_xgboost",
]:
    check(f"{fn_name} callable", lambda n=fn_name: (
        callable(getattr(sys.modules["src.model_training"], n))
    ))

# ── evaluation ────────────────────────────────
print("\n[ evaluation ]")
check("import module",
      lambda: __import__("src.evaluation", fromlist=["*"]))

for fn_name in [
    "compute_metrics", "evaluate_all_models", "overfitting_check",
    "plot_confusion_matrices", "plot_roc_curves",
    "plot_calibration_curves", "plot_feature_importances",
    "save_results_csv",
]:
    check(f"{fn_name} callable", lambda n=fn_name: (
        callable(getattr(sys.modules["src.evaluation"], n))
    ))

# ── shap_explainer ────────────────────────────
print("\n[ shap_explainer ]")
check("import module",
      lambda: __import__("src.shap_explainer", fromlist=["*"]))

check("SHAPExplainer is a class", lambda: (
    isinstance(sys.modules["src.shap_explainer"].SHAPExplainer, type)
))

for method in [
    "compute_shap_values", "plot_summary", "plot_feature_importance_bar",
    "plot_force", "plot_waterfall", "plot_dependence",
    "get_importance_table", "run_full_explanation",
]:
    check(f"SHAPExplainer.{method} exists", lambda m=method: (
        callable(getattr(sys.modules["src.shap_explainer"].SHAPExplainer, m))
    ))

# ── lime_explainer ────────────────────────────
print("\n[ lime_explainer ]")
check("import module",
      lambda: __import__("src.lime_explainer", fromlist=["*"]))

check("LIMEExplainer is a class", lambda: (
    isinstance(sys.modules["src.lime_explainer"].LIMEExplainer, type)
))

for method in [
    "explain_patient", "explain_multiple_patients",
    "run_stability_check", "global_feature_importance",
]:
    check(f"LIMEExplainer.{method} exists", lambda m=method: (
        callable(getattr(sys.modules["src.lime_explainer"].LIMEExplainer, m))
    ))

# ── third-party dependencies ──────────────────
print("\n[ third-party dependencies ]")
for pkg, import_name in [
    ("pandas",            "pandas"),
    ("numpy",             "numpy"),
    ("scikit-learn",      "sklearn"),
    ("imbalanced-learn",  "imblearn"),
    ("xgboost",           "xgboost"),
    ("shap",              "shap"),
    ("lime",              "lime"),
    ("matplotlib",        "matplotlib"),
    ("seaborn",           "seaborn"),
]:
    check(f"{pkg}", lambda m=import_name: __import__(m))

# ── Summary ───────────────────────────────────
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