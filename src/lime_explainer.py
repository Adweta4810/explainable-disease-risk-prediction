"""
lime_explainer.py
─────────────────
Reusable LIME explainability utilities for CKD and Diabetes pipelines.

Usage
-----
from src.lime_explainer import LIMEExplainer

# CKD — Random Forest
explainer = LIMEExplainer(
    model_pipeline = rf_best,              # fitted ImbPipeline
    X_train        = X_train,              # RAW training data (LIME needs distribution)
    X_test         = X_test,              # RAW test data
    feature_names  = feature_names,
    class_names    = ["Not CKD", "CKD"],
    output_dir     = "outputs/ckd/LIME",
    dataset_label  = "CKD",
)
explainer.explain_patient(patient_idx=0, y_test=y_test)
explainer.run_stability_check(patient_idx=0, n_runs=5)

# Diabetes — XGBoost  (pipeline handles its own scaling internally)
explainer = LIMEExplainer(
    model_pipeline = xgb_best,
    X_train        = X_train,
    X_test         = X_test,
    feature_names  = feature_names,
    class_names    = ["No Diabetes", "Diabetes"],
    output_dir     = "outputs/diabetes/LIME",
    dataset_label  = "Diabetes",
)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")


class LIMEExplainer:
    """
    Wraps LIME LimeTabularExplainer for classification pipelines.

    The fitted ImbPipeline is used as the prediction function, so any
    scaling or preprocessing inside the pipeline is applied automatically.

    Notes
    -----
    LIME explanations can vary between runs due to random sampling.
    Use run_stability_check() to assess explanation stability.
    """

    def __init__(
        self,
        model_pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: list[str],
        class_names: list[str] | None = None,
        output_dir: str = "outputs/lime",
        dataset_label: str = "Dataset",
        random_state: int = 42,
        discretize_continuous: bool = True,
    ):
        """
        Parameters
        ----------
        model_pipeline        : fitted ImbPipeline (predict_proba used internally)
        X_train               : raw training DataFrame (used to fit LIME kernel)
        X_test                : raw test DataFrame
        feature_names         : list of feature column names
        class_names           : [negative_label, positive_label]
        output_dir            : directory where plots and CSVs are saved
        dataset_label         : label used in titles and filenames
        random_state          : seed for LIME sampler
        discretize_continuous : passed to LimeTabularExplainer
        """
        self.pipeline      = model_pipeline
        self.X_train       = X_train
        self.X_test        = X_test
        self.feature_names = feature_names
        self.class_names   = class_names or ["Class 0", "Class 1"]
        self.output_dir    = output_dir
        self.label         = dataset_label
        self.random_state  = random_state

        os.makedirs(output_dir, exist_ok=True)

        # Build the LIME explainer once — fitted on training distribution
        self._lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=discretize_continuous,
            random_state=random_state,
        )
        print(f"[{self.label}] LIME explainer created.")
        print(f"  Classes     : {self.class_names}")
        print(f"  Features    : {len(feature_names)}")
        print(f"  Train rows  : {X_train.shape[0]}")

    # ── Single patient explanation ────────────────────────────────────────────

    def explain_patient(
        self,
        patient_idx: int = 0,
        y_test: pd.Series | None = None,
        num_features: int = 10,
        save_plot: bool = True,
        save_csv: bool = True,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Generate and save a LIME explanation for one patient.

        Parameters
        ----------
        patient_idx  : row index into X_test
        y_test       : true labels (used only for display in title)
        num_features : number of features to include in explanation
        save_plot    : save explanation bar chart as PNG
        save_csv     : save contribution table as CSV
        show_plot    : display plot interactively

        Returns
        -------
        pd.DataFrame with columns [Feature condition, Contribution, Effect]
        """
        patient_data = self.X_test.iloc[patient_idx]
        pred_label   = self.pipeline.predict(self.X_test.iloc[[patient_idx]])[0]
        pred_prob    = self.pipeline.predict_proba(self.X_test.iloc[[patient_idx]])[0, 1]

        # Patient info summary
        print(f"\n[{self.label}] Patient {patient_idx}:")
        if y_test is not None:
            true_lbl = y_test.iloc[patient_idx]
            print(f"  True label      : {self.class_names[int(true_lbl)]} ({true_lbl})")
        print(f"  Predicted label : {self.class_names[int(pred_label)]} ({pred_label})")
        print(f"  P(positive)     : {pred_prob:.4f}")

        # Generate explanation
        lime_exp = self._lime_explainer.explain_instance(
            data_row=patient_data.values,
            predict_fn=self.pipeline.predict_proba,
            num_features=num_features,
        )

        # ── Plot ──────────────────────────────────────────────────────────────
        title_parts = [f"LIME — {self.label} Patient {patient_idx}"]
        if y_test is not None:
            title_parts.append(f"True: {self.class_names[int(true_lbl)]}")
        title_parts.append(
            f"Pred: {self.class_names[int(pred_label)]}  P={pred_prob:.3f}"
        )
        title = "  |  ".join(title_parts)

        fig = lime_exp.as_pyplot_figure()
        plt.title(title, fontsize=10, pad=10)
        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(
                self.output_dir,
                f"{self.label.lower()}_lime_explanation_patient{patient_idx}.png",
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"  Plot saved → {plot_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        # ── Contribution table ────────────────────────────────────────────────
        table = pd.DataFrame(
            lime_exp.as_list(),
            columns=["Feature condition", "Contribution"],
        )
        table["Effect"] = np.where(
            table["Contribution"] > 0,
            f"Supports {self.class_names[1]} prediction",
            f"Supports {self.class_names[0]} prediction",
        )
        table["Abs Contribution"] = table["Contribution"].abs()
        table = table.sort_values("Abs Contribution", ascending=False).reset_index(drop=True)

        print(f"\n  Top local feature contributions (Patient {patient_idx}):")
        print(table[["Feature condition", "Contribution", "Effect"]].to_string(index=False))

        if save_csv:
            csv_path = os.path.join(
                self.output_dir,
                f"{self.label.lower()}_lime_explanation_patient{patient_idx}.csv",
            )
            table.to_csv(csv_path, index=False)
            print(f"  CSV saved  → {csv_path}")

        return table

    # ── Stability check ───────────────────────────────────────────────────────

    def run_stability_check(
        self,
        patient_idx: int = 0,
        n_runs: int = 5,
        num_features: int = 10,
        save_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Re-run LIME n_runs times with different random seeds and measure
        explanation stability (variance of contribution scores).

        LIME explanations can vary due to random neighbourhood sampling
        (Garreau & von Luxburg, 2020; Zhang et al., 2019).

        Parameters
        ----------
        patient_idx  : row index into X_test
        n_runs       : number of independent LIME runs
        num_features : features per explanation
        save_plot    : save stability box-plot as PNG

        Returns
        -------
        pd.DataFrame: mean, std, cv (coefficient of variation) per feature condition.
        """
        print(f"\n[{self.label}] LIME Stability Check — Patient {patient_idx} ({n_runs} runs)")

        patient_data = self.X_test.iloc[patient_idx]
        all_results  = []

        for run in range(n_runs):
            lime_run = LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode="classification",
                discretize_continuous=True,
                random_state=self.random_state + run,   # different seed each run
            )
            exp = lime_run.explain_instance(
                data_row=patient_data.values,
                predict_fn=self.pipeline.predict_proba,
                num_features=num_features,
            )
            run_dict = {cond: val for cond, val in exp.as_list()}
            run_dict["run"] = run
            all_results.append(run_dict)

        stability_df  = pd.DataFrame(all_results).set_index("run")
        feature_cols  = [c for c in stability_df.columns]

        mean_vals = stability_df[feature_cols].mean()
        std_vals  = stability_df[feature_cols].std()
        cv_vals   = (std_vals / mean_vals.abs().replace(0, np.nan)).abs()

        summary = pd.DataFrame({
            "Feature condition": feature_cols,
            "Mean contribution": mean_vals.values,
            "Std":               std_vals.values,
            "CV (%)":            (cv_vals * 100).values,
        }).sort_values("Mean contribution", key=abs, ascending=False).reset_index(drop=True)

        print("\n  Stability summary:")
        print(summary.round(4).to_string(index=False))

        # ── Box plot ──────────────────────────────────────────────────────────
        if save_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            stability_df[feature_cols].boxplot(ax=ax, vert=False)
            ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Zero")
            ax.set_title(
                f"LIME Stability — {self.label} Patient {patient_idx} "
                f"({n_runs} runs, different random seeds)",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel("LIME Contribution")
            ax.legend()
            plt.tight_layout()

            plot_path = os.path.join(
                self.output_dir,
                f"{self.label.lower()}_lime_stability_patient{patient_idx}.png",
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.show()
            print(f"  Stability plot saved → {plot_path}")

        return summary

    # ── Batch explanation ─────────────────────────────────────────────────────

    def explain_multiple_patients(
        self,
        patient_indices: list[int],
        y_test: pd.Series | None = None,
        num_features: int = 10,
    ) -> dict:
        """
        Generate LIME explanations for a list of patient indices.

        Returns
        -------
        dict mapping patient_idx → contribution DataFrame
        """
        results = {}
        for idx in patient_indices:
            print(f"\n{'─'*50}")
            results[idx] = self.explain_patient(
                patient_idx=idx,
                y_test=y_test,
                num_features=num_features,
                show_plot=True,
            )
        return results

    # ── Global pseudo-importance ──────────────────────────────────────────────

    def global_feature_importance(
        self,
        n_patients: int = 50,
        num_features: int = 10,
        save_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Approximate global feature importance by averaging |LIME contributions|
        across n_patients randomly sampled from X_test.

        Note: This is an approximation — LIME is inherently local.
        Use SHAP for true global importance.
        """
        print(f"\n[{self.label}] LIME global importance ({n_patients} patients) …")

        n = min(n_patients, len(self.X_test))
        indices = np.random.RandomState(self.random_state).choice(
            len(self.X_test), size=n, replace=False
        )

        accum = {}
        for idx in indices:
            exp = self._lime_explainer.explain_instance(
                data_row=self.X_test.iloc[idx].values,
                predict_fn=self.pipeline.predict_proba,
                num_features=num_features,
            )
            for cond, val in exp.as_list():
                accum.setdefault(cond, []).append(abs(val))

        importance = pd.DataFrame({
            "Feature condition": list(accum.keys()),
            "Mean |LIME|":       [np.mean(v) for v in accum.values()],
        }).sort_values("Mean |LIME|", ascending=False).reset_index(drop=True)

        print(importance.head(15).to_string(index=False))

        if save_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            importance.head(15).set_index("Feature condition")["Mean |LIME|"] \
                .sort_values().plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(
                f"LIME Global Feature Importance — {self.label}\n"
                f"(Mean |contribution| over {n} patients)",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel("Mean |LIME Contribution|")
            plt.tight_layout()

            plot_path = os.path.join(
                self.output_dir,
                f"{self.label.lower()}_lime_global_importance.png",
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.show()
            print(f"  Plot saved → {plot_path}")

        return importance
