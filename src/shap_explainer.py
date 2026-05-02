"""
shap_explainer.py
─────────────────
Reusable SHAP explainability utilities for CKD (Random Forest)
and Diabetes (XGBoost) pipelines.

Usage
-----
from src.shap_explainer import SHAPExplainer

# CKD — Random Forest
explainer = SHAPExplainer(
    model_pipeline = rf_best,          # fitted ImbPipeline
    X_test         = X_test,           # RAW unscaled test data
    feature_names  = feature_names,
    class_names    = ["Not CKD", "CKD"],
    output_dir     = "outputs/ckd/SHAP",
    dataset_label  = "CKD",
)
explainer.compute_shap_values()
explainer.plot_summary()
explainer.plot_feature_importance_bar()
explainer.plot_force(patient_idx=0, y_test=y_test)
explainer.plot_waterfall(patient_idx=0, y_test=y_test)  # XGBoost only

# Diabetes — XGBoost  (pass scaled test data, or let the class handle it)
explainer = SHAPExplainer(
    model_pipeline = xgb_best,
    X_test         = X_test,           # RAW; scaler is extracted from pipeline
    feature_names  = feature_names,
    class_names    = ["No Diabetes", "Diabetes"],
    output_dir     = "outputs/diabetes/SHAP",
    dataset_label  = "Diabetes",
    scale_for_shap = True,             # extract scaler from pipeline and apply
)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# initjs() enables interactive force plots in Jupyter.
# Only call it when IPython is available (i.e. inside a notebook).
try:
    import IPython
    shap.initjs()
except Exception:
    pass


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for pipelines that include a tree-based classifier.

    Attributes
    ----------
    shap_values_pos : np.ndarray
        SHAP values for the positive class, shape (n_samples, n_features).
    expected_value_pos : float
        Base value (expected model output) for the positive class.
    X_explain : pd.DataFrame
        The data passed to SHAP (scaled if scale_for_shap=True).
    """

    def __init__(
        self,
        model_pipeline,
        X_test: pd.DataFrame,
        feature_names: list[str],
        class_names: list[str] | None = None,
        output_dir: str = "outputs/shap",
        dataset_label: str = "Dataset",
        scale_for_shap: bool = False,
    ):
        """
        Parameters
        ----------
        model_pipeline : fitted ImbPipeline
        X_test         : raw (unscaled) test DataFrame
        feature_names  : list of column names
        class_names    : [negative_label, positive_label]
        output_dir     : directory where plots are saved
        dataset_label  : label used in plot titles and filenames
        scale_for_shap : if True, the scaler inside the pipeline is extracted
                         and applied to X_test before computing SHAP values.
                         Required for XGBoost pipelines that include a scaler.
        """
        self.pipeline     = model_pipeline
        self.feature_names = feature_names
        self.class_names  = class_names or ["Class 0", "Class 1"]
        self.output_dir   = output_dir
        self.label        = dataset_label
        self.scale        = scale_for_shap

        os.makedirs(output_dir, exist_ok=True)

        # Pull the raw classifier out of the pipeline
        self.classifier = model_pipeline.named_steps["classifier"]

        # Prepare the data matrix SHAP will receive
        if scale_for_shap and "scaler" in model_pipeline.named_steps:
            scaler = model_pipeline.named_steps["scaler"]
            self.X_explain = pd.DataFrame(
                scaler.transform(X_test),
                columns=feature_names,
                index=X_test.index,
            )
        else:
            self.X_explain = X_test.copy()
            self.X_explain.columns = feature_names

        self.shap_values_pos    = None
        self.expected_value_pos = None
        self._raw_shap          = None

    # ── Compute ───────────────────────────────────────────────────────────────

    def compute_shap_values(self) -> None:
        """Compute SHAP values using TreeExplainer. Must be called first."""
        print(f"[{self.label}] Computing SHAP values …")
        explainer        = shap.TreeExplainer(self.classifier)
        self._raw_shap   = explainer.shap_values(self.X_explain)
        self._explainer  = explainer

        # Normalise across SHAP library versions
        if isinstance(self._raw_shap, list):
            # Multi-output list: [class0_array, class1_array]
            self.shap_values_pos    = self._raw_shap[1]
            self.expected_value_pos = explainer.expected_value[1]
        elif self._raw_shap.ndim == 3:
            # 3-D array: (n_samples, n_features, n_classes)
            self.shap_values_pos    = self._raw_shap[:, :, 1]
            self.expected_value_pos = explainer.expected_value[1]
        else:
            # Binary XGBoost: single 2-D array
            self.shap_values_pos    = self._raw_shap
            self.expected_value_pos = explainer.expected_value

        print(f"  SHAP values shape  : {self.shap_values_pos.shape}")
        print(f"  Expected value (+) : {self.expected_value_pos:.4f}")

    def _check_computed(self) -> None:
        if self.shap_values_pos is None:
            raise RuntimeError("Call compute_shap_values() first.")

    # ── Global plots ──────────────────────────────────────────────────────────

    def plot_summary(
        self,
        max_display: int = 20,
        filename: str | None = None,
    ) -> None:
        """Beeswarm summary plot — shows direction and magnitude of feature impact."""
        self._check_computed()

        fname = filename or f"{self.label.lower()}_shap_summary_dot.png"
        save_path = os.path.join(self.output_dir, fname)

        plt.figure()
        shap.summary_plot(
            self.shap_values_pos,
            self.X_explain,
            max_display=max_display,
            show=False,
        )
        plt.title(
            f"SHAP Summary Plot — {self.label} ({type(self.classifier).__name__})",
            fontsize=12, fontweight="bold", pad=12,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")

    def plot_feature_importance_bar(
        self,
        max_display: int = 20,
        filename: str | None = None,
    ) -> None:
        """Bar chart of mean |SHAP| per feature — ranked global importance."""
        self._check_computed()

        fname = filename or f"{self.label.lower()}_shap_feature_importance_bar.png"
        save_path = os.path.join(self.output_dir, fname)

        plt.figure()
        shap.summary_plot(
            self.shap_values_pos,
            self.X_explain,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title(
            f"SHAP Feature Importance (Mean |SHAP|) — {self.label}",
            fontsize=12, fontweight="bold", pad=12,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")

        # Print ranked table
        mean_shap = pd.Series(
            np.abs(self.shap_values_pos).mean(axis=0),
            index=self.feature_names,
        ).sort_values(ascending=False)

        print(f"\nRanked Feature Importance (Mean |SHAP|) — {self.label}:")
        print("-" * 38)
        for rank, (feat, val) in enumerate(mean_shap.items(), 1):
            print(f"  {rank:>2}. {feat:<20}  {val:.4f}")

    def get_importance_table(self) -> pd.DataFrame:
        """Return a DataFrame of mean |SHAP| values, sorted descending."""
        self._check_computed()
        return pd.DataFrame({
            "Feature":   self.feature_names,
            "Mean|SHAP|": np.abs(self.shap_values_pos).mean(axis=0),
        }).sort_values("Mean|SHAP|", ascending=False).reset_index(drop=True)

    # ── Local plots ───────────────────────────────────────────────────────────

    def plot_force(
        self,
        patient_idx: int = 0,
        y_test: pd.Series | None = None,
        y_pred: np.ndarray | None = None,
        filename: str | None = None,
    ) -> None:
        """
        Static force plot for a single patient.
        Shows which features pushed the prediction above/below the base value.
        """
        self._check_computed()

        fname = filename or f"{self.label.lower()}_shap_force_plot_patient{patient_idx}.png"
        save_path = os.path.join(self.output_dir, fname)

        # Build a human-readable title
        title_parts = [f"SHAP Force Plot — {self.label} Patient {patient_idx}"]
        if y_test is not None:
            true_lbl = y_test.iloc[patient_idx]
            title_parts.append(
                f"True: {self.class_names[int(true_lbl)]}"
            )
        if y_pred is not None:
            pred_lbl = y_pred[patient_idx]
            pred_prob = self.pipeline.predict_proba(
                self.X_explain.iloc[[patient_idx]]
            )[0, 1]
            title_parts.append(
                f"Pred: {self.class_names[int(pred_lbl)]}  P={pred_prob:.3f}"
            )
        title = "  |  ".join(title_parts)

        plt.figure()
        shap.force_plot(
            self.expected_value_pos,
            self.shap_values_pos[patient_idx],
            self.X_explain.iloc[patient_idx],
            matplotlib=True,
            show=False,
            figsize=(16, 3),
        )
        plt.title(title, fontsize=9, pad=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")

    def plot_waterfall(
        self,
        patient_idx: int = 0,
        y_test: pd.Series | None = None,
        y_pred: np.ndarray | None = None,
        max_display: int = 12,
        filename: str | None = None,
    ) -> None:
        """
        Waterfall plot for a single patient.
        Best suited for XGBoost (single expected_value scalar).
        """
        self._check_computed()

        fname = filename or f"{self.label.lower()}_shap_waterfall_patient{patient_idx}.png"
        save_path = os.path.join(self.output_dir, fname)

        explanation = shap.Explanation(
            values=self.shap_values_pos[patient_idx],
            base_values=self.expected_value_pos,
            data=self.X_explain.iloc[patient_idx].values,
            feature_names=self.feature_names,
        )

        title_parts = [f"SHAP Waterfall — {self.label} Patient {patient_idx}"]
        if y_test is not None:
            true_lbl = y_test.iloc[patient_idx]
            title_parts.append(f"True: {self.class_names[int(true_lbl)]}")
        if y_pred is not None:
            pred_lbl  = y_pred[patient_idx]
            pred_prob = self.pipeline.predict_proba(
                self.X_explain.iloc[[patient_idx]]
            )[0, 1]
            title_parts.append(
                f"Pred: {self.class_names[int(pred_lbl)]}  P={pred_prob:.3f}"
            )

        plt.figure()
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title("  |  ".join(title_parts), fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")

    def plot_dependence(
        self,
        feature: str | None = None,
        filename: str | None = None,
    ) -> None:
        """
        Dependence plot for the most important feature (or a specified one).
        Shows how a feature's value correlates with its SHAP impact.
        """
        self._check_computed()

        if feature is None:
            mean_abs = np.abs(self.shap_values_pos).mean(axis=0)
            feature  = self.feature_names[int(np.argmax(mean_abs))]
            print(f"[{self.label}] Most important feature: {feature}")

        fname = filename or f"{self.label.lower()}_shap_dependence_{feature}.png"
        save_path = os.path.join(self.output_dir, fname)

        plt.figure()
        shap.dependence_plot(
            feature,
            self.shap_values_pos,
            self.X_explain,
            show=False,
        )
        plt.title(f"SHAP Dependence Plot — {feature} ({self.label})", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")

    # ── Convenience: run all standard plots ───────────────────────────────────

    def run_full_explanation(
        self,
        patient_idx: int = 0,
        y_test: pd.Series | None = None,
        y_pred: np.ndarray | None = None,
    ) -> None:
        """
        Compute SHAP values and generate all standard plots in one call.
        """
        self.compute_shap_values()
        self.plot_summary()
        self.plot_feature_importance_bar()
        self.plot_force(patient_idx=patient_idx, y_test=y_test, y_pred=y_pred)
        self.plot_waterfall(patient_idx=patient_idx, y_test=y_test, y_pred=y_pred)
        self.plot_dependence()
        print(f"\n[{self.label}] All SHAP plots saved to: {self.output_dir}")
