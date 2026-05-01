"""
Disease Risk Prediction Dashboard
Explainable ML for CKD & Diabetes
Author: Adweta Sigdel
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
from lime import lime_tabular
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #a0cfff !important; }

    /* Header banner */
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: white;
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 22px;
        border-left: 5px solid #e94560;
    }
    .main-header h1 { margin: 0; font-size: 1.75rem; }
    .main-header p  { margin: 6px 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .risk-high   { background: #ffe5e5; color: #c0392b; border: 2px solid #e74c3c; }
    .risk-medium { background: #fff3e0; color: #e67e22; border: 2px solid #f39c12; }
    .risk-low    { background: #e8f8f0; color: #27ae60; border: 2px solid #2ecc71; }

    /* Metric card */
    .metric-card {
        background: #f7f9fc;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        border: 1px solid #e0e6ed;
    }
    .metric-card .label { font-size: 0.78rem; color: #6c757d; font-weight: 600;
                          text-transform: uppercase; letter-spacing: 0.6px; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; margin-top: 4px; }

    /* Section divider */
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #0f3460;
        border-bottom: 2px solid #e94560; padding-bottom: 5px;
        margin: 22px 0 14px;
    }

    /* Info box */
    .info-box {
        background: #eef4ff; border-left: 4px solid #3b82f6;
        padding: 10px 14px; border-radius: 6px; font-size: 0.88rem;
        color: #1e3a8a; margin: 10px 0;
    }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────────────────────
CKD_FEATURES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn",
    "dm", "cad", "appet", "pe", "ane"
]

CKD_LABELS = {
    "age": "Age (years)", "bp": "Blood Pressure (mm/Hg)",
    "sg": "Specific Gravity", "al": "Albumin (0–5)",
    "su": "Sugar (0–5)", "rbc": "Red Blood Cells",
    "pc": "Pus Cell", "pcc": "Pus Cell Clumps",
    "ba": "Bacteria", "bgr": "Blood Glucose Random (mg/dl)",
    "bu": "Blood Urea (mg/dl)", "sc": "Serum Creatinine (mg/dl)",
    "sod": "Sodium (mEq/L)", "pot": "Potassium (mEq/L)",
    "hemo": "Haemoglobin (g/dl)", "pcv": "Packed Cell Volume",
    "wbcc": "WBC Count (cells/cumm)", "rbcc": "RBC Count (millions/cumm)",
    "htn": "Hypertension", "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease", "appet": "Appetite",
    "pe": "Pedal Edema", "ane": "Anaemia"
}

CKD_CATEGORICAL = {
    "rbc": ["normal", "abnormal"], "pc": ["normal", "abnormal"],
    "pcc": ["notpresent", "present"], "ba": ["notpresent", "present"],
    "htn": ["no", "yes"], "dm": ["no", "yes"],
    "cad": ["no", "yes"], "appet": ["good", "poor"],
    "pe": ["no", "yes"], "ane": ["no", "yes"]
}

CKD_RANGES = {
    "age": (1, 100, 50), "bp": (50, 180, 80), "sg": (1.005, 1.025, 1.020),
    "al": (0, 5, 0), "su": (0, 5, 0), "bgr": (22, 490, 100),
    "bu": (1.5, 391.0, 30.0), "sc": (0.4, 76.0, 1.0),
    "sod": (4.5, 163.0, 137.0), "pot": (2.5, 47.0, 4.5),
    "hemo": (3.1, 17.8, 13.0), "pcv": (9, 54, 41),
    "wbcc": (2200, 26400, 7500), "rbcc": (2.1, 8.0, 4.5)
}

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

DIABETES_LABELS = {
    "Pregnancies": "Pregnancies", "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)", "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin (μU/mL)", "BMI": "BMI (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function", "Age": "Age (years)"
}

DIABETES_RANGES = {
    "Pregnancies": (0, 17, 3), "Glucose": (44, 200, 120),
    "BloodPressure": (24, 122, 70), "SkinThickness": (7, 99, 23),
    "Insulin": (14, 846, 80), "BMI": (18.0, 67.1, 32.0),
    "DiabetesPedigreeFunction": (0.078, 2.42, 0.47), "Age": (21, 81, 33)
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    paths = {
        "ckd_xgb":      "outputs/xgb_pipeline.pkl",
        "ckd_rf":       "outputs/rf_pipeline.pkl",
        "ckd_lr":       "outputs/lr_pipeline.pkl",
        "diabetes_xgb": "outputs/xgb_diabetes_pipeline.pkl",
        "diabetes_rf":  "outputs/rf_diabetes_pipeline.pkl",
        "diabetes_lr":  "outputs/lr_diabetes_pipeline.pkl",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models


@st.cache_data(show_spinner=False)
def load_training_data():
    data = {}
    ckd_path = "../data/chronic_kidney_disease/ckd_cleaned.csv"
    dia_path  = "../data/diabetes/diabetes_cleaned.csv"
    if os.path.exists(ckd_path):
        df = pd.read_csv(ckd_path)
        data["ckd_X"] = df.drop(columns=["classification"])
        data["ckd_y"] = df["classification"]
    if os.path.exists(dia_path):
        df = pd.read_csv(dia_path)
        data["dia_X"] = df.drop(columns=["Outcome"])
        data["dia_y"] = df["Outcome"]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Risk level helper
# ─────────────────────────────────────────────────────────────────────────────
def risk_level(prob):
    if prob >= 0.65:
        return "HIGH", "risk-high", "🔴"
    elif prob >= 0.35:
        return "MEDIUM", "risk-medium", "🟡"
    else:
        return "LOW", "risk-low", "🟢"


# ─────────────────────────────────────────────────────────────────────────────
# SHAP explanation plot
# ─────────────────────────────────────────────────────────────────────────────
def shap_waterfall(pipeline, input_df, feature_names, title):
    try:
        clf = pipeline.named_steps["classifier"]
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(input_df)

        exp = shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.waterfall_plot(exp, max_display=12, show=False)
        plt.title(title, fontsize=12, pad=10)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None


def shap_bar_global(pipeline, X_train, feature_names, title):
    try:
        clf = pipeline.named_steps["classifier"]
        sample = X_train.sample(min(200, len(X_train)), random_state=42)
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(sample)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=True).tail(12)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e94560" if v > importance.median() else "#3b82f6" for v in importance.values]
        importance.plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LIME explanation plot
# ─────────────────────────────────────────────────────────────────────────────
def lime_explanation(pipeline, X_train, input_row, feature_names, class_names, title):
    try:
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            random_state=42
        )
        exp = explainer.explain_instance(
            data_row=input_row,
            predict_fn=pipeline.predict_proba,
            num_features=12,
            num_samples=3000
        )
        pairs = sorted(exp.as_list(label=1), key=lambda x: abs(x[1]), reverse=True)
        feats  = [p[0] for p in pairs]
        weights = [p[1] for p in pairs]
        colors = ["#e94560" if w > 0 else "#3b82f6" for w in weights]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(feats[::-1], weights[::-1], color=colors[::-1],
                edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("LIME Weight  (positive → higher risk)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        red_p  = mpatches.Patch(color="#e94560", label=f"↑ {class_names[1]} risk")
        blue_p = mpatches.Patch(color="#3b82f6", label=f"↓ {class_names[1]} risk")
        ax.legend(handles=[red_p, blue_p], fontsize=8, loc="lower right")
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LIME Stability plot
# ─────────────────────────────────────────────────────────────────────────────
def lime_stability(pipeline, X_train, input_row, feature_names, class_names, n_runs=10):
    try:
        all_weights = []
        for seed in range(n_runs):
            exp_obj = lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=feature_names,
                class_names=class_names,
                mode="classification",
                random_state=seed
            ).explain_instance(
                data_row=input_row,
                predict_fn=pipeline.predict_proba,
                num_features=len(feature_names),
                num_samples=3000
            )
            all_weights.append(dict(exp_obj.as_list(label=1)))

        df_w = pd.DataFrame(all_weights).fillna(0)
        means = df_w.mean().sort_values(key=abs, ascending=False).head(12)
        stds  = df_w.std()[means.index]

        colors = ["#e94560" if m > 0 else "#3b82f6" for m in means.values]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            range(len(means)), means.values,
            xerr=stds.values, color=colors,
            edgecolor="white", linewidth=0.3,
            capsize=4, error_kw={"elinewidth": 1.1, "ecolor": "black"}
        )
        ax.set_yticks(range(len(means)))
        ax.set_yticklabels(means.index, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean LIME weight  (± std, 10 runs)", fontsize=10)
        ax.set_title(
            f"LIME Stability ({n_runs} runs) — error bars show sampling variance",
            fontsize=11, pad=8
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Disease Risk Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        ["🏠 Home", "🫘 CKD Prediction", "🩸 Diabetes Prediction", "📊 Model Comparison"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown(
        "1. Select a disease tab\n"
        "2. Enter patient values\n"
        "3. Click **Predict**\n"
        "4. View risk level, SHAP & LIME explanations"
    )
    st.markdown("---")
    st.markdown(
        "<small>BSc Computing Final Project · Adweta Sigdel<br>"
        "Supervisor: Dr. Hari Joshi</small>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    models = load_models()
    train_data = load_training_data()

no_models = len(models) == 0


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Explainable ML for Disease Risk Prediction</h1>
        <p>Predict CKD & Diabetes risk with clear, patient-level explanations using SHAP and LIME</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="label">Diseases Covered</div>
            <div class="value">2</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="label">ML Models</div>
            <div class="value">3</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="label">XAI Methods</div>
            <div class="value">2</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🫘 Chronic Kidney Disease")
        st.markdown(
            "Predicts CKD risk from 24 clinical features including "
            "haemoglobin, serum creatinine, albumin, blood pressure and more."
        )
        st.markdown("**Best model:** XGBoost (SMOTE + tuning)")
    with c2:
        st.markdown("### 🩸 Diabetes")
        st.markdown(
            "Predicts diabetes risk from 8 features: glucose, BMI, "
            "age, insulin, blood pressure, skin thickness and pedigree function."
        )
        st.markdown("**Best model:** XGBoost (SMOTE + tuning)")

    st.markdown("---")
    st.markdown("### 🔍 Explainability Methods")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **SHAP** *(SHapley Additive exPlanations)*
        - Based on cooperative game theory
        - Exact, consistent, globally & locally applicable
        - Waterfall plots show per-feature contribution for each patient
        """)
    with c2:
        st.markdown("""
        **LIME** *(Local Interpretable Model-agnostic Explanations)*
        - Fits a local linear model around each prediction
        - Model-agnostic — works with any classifier
        - Stability check included (10 repeated runs)
        """)

    if no_models:
        st.warning(
            "⚠️ No model files found. Please train models in Notebooks 02 & 04 "
            "and ensure `outputs/*.pkl` files exist in the same directory as this app."
        )


# ─────────────────────────────────────────────────────────────────────────────
# CKD PREDICTION PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🫘 CKD Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🫘 Chronic Kidney Disease Risk Prediction</h1>
        <p>Enter patient clinical values to predict CKD risk with SHAP & LIME explanations</p>
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    ckd_model_key = st.selectbox(
        "Select model",
        options=["ckd_xgb", "ckd_rf", "ckd_lr"],
        format_func=lambda x: {"ckd_xgb": "XGBoost (Best)", "ckd_rf": "Random Forest", "ckd_lr": "Logistic Regression"}[x]
    )
    ckd_pipeline = models.get(ckd_model_key)

    st.markdown('<div class="section-title">Patient Input Features</div>', unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("ckd_form"):
        col_a, col_b, col_c = st.columns(3)

        def num_input(col, key):
            lo, hi, default = CKD_RANGES.get(key, (0, 100, 50))
            return col.number_input(CKD_LABELS[key], min_value=float(lo),
                                    max_value=float(hi), value=float(default), key=key)

        def cat_input(col, key):
            opts = CKD_CATEGORICAL[key]
            choice = col.selectbox(CKD_LABELS[key], opts, key=key)
            return opts.index(choice)

        with col_a:
            age   = num_input(col_a, "age")
            bp    = num_input(col_a, "bp")
            sg    = num_input(col_a, "sg")
            al    = num_input(col_a, "al")
            su    = num_input(col_a, "su")
            bgr   = num_input(col_a, "bgr")
            bu    = num_input(col_a, "bu")
            sc    = num_input(col_a, "sc")

        with col_b:
            sod   = num_input(col_b, "sod")
            pot   = num_input(col_b, "pot")
            hemo  = num_input(col_b, "hemo")
            pcv   = num_input(col_b, "pcv")
            wbcc  = num_input(col_b, "wbcc")
            rbcc  = num_input(col_b, "rbcc")
            rbc   = cat_input(col_b, "rbc")
            pc    = cat_input(col_b, "pc")

        with col_c:
            pcc   = cat_input(col_c, "pcc")
            ba    = cat_input(col_c, "ba")
            htn   = cat_input(col_c, "htn")
            dm    = cat_input(col_c, "dm")
            cad   = cat_input(col_c, "cad")
            appet = cat_input(col_c, "appet")
            pe    = cat_input(col_c, "pe")
            ane   = cat_input(col_c, "ane")

        submitted = st.form_submit_button("🔍 Predict CKD Risk", use_container_width=True)

    if submitted:
        input_values = [age, bp, sg, al, su, rbc, pc, pcc, ba,
                        bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc,
                        htn, dm, cad, appet, pe, ane]
        input_df = pd.DataFrame([input_values], columns=CKD_FEATURES)

        if ckd_pipeline is None:
            st.error("CKD model not loaded. Please check model files.")
        else:
            prob = ckd_pipeline.predict_proba(input_df)[0][1]
            level, badge_class, emoji = risk_level(prob)

            # ── Result banner ─────────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns([1, 1, 1])
            with r1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">CKD Risk Probability</div>
                    <div class="value">{prob:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Risk Level</div>
                    <div class="value"><span class="risk-badge {badge_class}">{emoji} {level}</span></div>
                </div>""", unsafe_allow_html=True)
            with r3:
                inv = 1 - prob
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Healthy Probability</div>
                    <div class="value">{inv:.1%}</div>
                </div>""", unsafe_allow_html=True)

            st.progress(float(prob), text=f"CKD Risk: {prob:.1%}")

            # ── Explainability tabs ───────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-box">The charts below explain <strong>why</strong> this prediction was made. '
                'Red features increase CKD risk; blue features decrease it.</div>',
                unsafe_allow_html=True
            )

            tab_shap, tab_lime, tab_stab = st.tabs(
                ["🟠 SHAP (Patient)", "🟣 LIME (Patient)", "📉 LIME Stability"]
            )

            with tab_shap:
                st.markdown("**SHAP Waterfall** — contribution of each feature for this patient")
                fig = shap_waterfall(
                    ckd_pipeline, input_df, CKD_FEATURES,
                    f"SHAP Waterfall — CKD Prediction (prob={prob:.3f})"
                )
                if fig:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.warning("SHAP explanation unavailable for this model.")

                if "ckd_X" in train_data:
                    st.markdown("**SHAP Global Importance** — average feature importance across training data")
                    fig2 = shap_bar_global(
                        ckd_pipeline, train_data["ckd_X"], CKD_FEATURES,
                        "Mean |SHAP| — CKD Feature Importance (XGBoost)"
                    )
                    if fig2:
                        st.pyplot(fig2, use_container_width=True)

            with tab_lime:
                st.markdown("**LIME Explanation** — local linear approximation for this patient")
                if "ckd_X" in train_data:
                    fig = lime_explanation(
                        ckd_pipeline, train_data["ckd_X"], input_values,
                        CKD_FEATURES, ["Not CKD", "CKD"],
                        f"LIME — CKD Patient (prob={prob:.3f})"
                    )
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("LIME explanation unavailable.")
                else:
                    st.info("Training data not found. LIME needs training data — check data path.")

            with tab_stab:
                st.markdown(
                    "**LIME Stability Check** — 10 repeated runs with different random seeds.  \n"
                    "Narrow error bars = stable explanations. Wide bars = sampling variance "
                    "*(Garreau & von Luxburg, 2020)*."
                )
                if "ckd_X" in train_data:
                    with st.spinner("Running 10 LIME iterations…"):
                        fig = lime_stability(
                            ckd_pipeline, train_data["ckd_X"], input_values,
                            CKD_FEATURES, ["Not CKD", "CKD"]
                        )
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Training data not found.")


# ─────────────────────────────────────────────────────────────────────────────
# DIABETES PREDICTION PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🩸 Diabetes Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🩸 Diabetes Risk Prediction</h1>
        <p>Enter patient values to predict diabetes risk with SHAP & LIME explanations</p>
    </div>
    """, unsafe_allow_html=True)

    dia_model_key = st.selectbox(
        "Select model",
        options=["diabetes_xgb", "diabetes_rf", "diabetes_lr"],
        format_func=lambda x: {
            "diabetes_xgb": "XGBoost (Best)",
            "diabetes_rf":  "Random Forest",
            "diabetes_lr":  "Logistic Regression"
        }[x]
    )
    dia_pipeline = models.get(dia_model_key)

    st.markdown('<div class="section-title">Patient Input Features</div>', unsafe_allow_html=True)

    with st.form("dia_form"):
        col_a, col_b = st.columns(2)

        def dia_num(col, key):
            lo, hi, default = DIABETES_RANGES[key]
            step = 0.001 if key == "DiabetesPedigreeFunction" else (0.1 if key == "BMI" else 1.0)
            return col.number_input(
                DIABETES_LABELS[key],
                min_value=float(lo), max_value=float(hi),
                value=float(default), step=step, key="dia_" + key
            )

        with col_a:
            preg = dia_num(col_a, "Pregnancies")
            gluc = dia_num(col_a, "Glucose")
            bp   = dia_num(col_a, "BloodPressure")
            skin = dia_num(col_a, "SkinThickness")

        with col_b:
            ins  = dia_num(col_b, "Insulin")
            bmi  = dia_num(col_b, "BMI")
            dpf  = dia_num(col_b, "DiabetesPedigreeFunction")
            age  = dia_num(col_b, "Age")

        submitted_dia = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

    if submitted_dia:
        input_values_dia = [preg, gluc, bp, skin, ins, bmi, dpf, age]
        input_df_dia = pd.DataFrame([input_values_dia], columns=DIABETES_FEATURES)

        if dia_pipeline is None:
            st.error("Diabetes model not loaded. Please check model files.")
        else:
            prob = dia_pipeline.predict_proba(input_df_dia)[0][1]
            level, badge_class, emoji = risk_level(prob)

            st.markdown("---")
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Diabetes Risk Probability</div>
                    <div class="value">{prob:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Risk Level</div>
                    <div class="value"><span class="risk-badge {badge_class}">{emoji} {level}</span></div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Healthy Probability</div>
                    <div class="value">{(1-prob):.1%}</div>
                </div>""", unsafe_allow_html=True)

            st.progress(float(prob), text=f"Diabetes Risk: {prob:.1%}")

            st.markdown("---")
            st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-box">Red features increase diabetes risk; '
                'blue features decrease it.</div>',
                unsafe_allow_html=True
            )

            tab_shap, tab_lime, tab_stab = st.tabs(
                ["🟠 SHAP (Patient)", "🟣 LIME (Patient)", "📉 LIME Stability"]
            )

            with tab_shap:
                st.markdown("**SHAP Waterfall** — feature contributions for this patient")
                fig = shap_waterfall(
                    dia_pipeline, input_df_dia, DIABETES_FEATURES,
                    f"SHAP Waterfall — Diabetes Prediction (prob={prob:.3f})"
                )
                if fig:
                    st.pyplot(fig, use_container_width=True)

                if "dia_X" in train_data:
                    st.markdown("**SHAP Global Importance**")
                    fig2 = shap_bar_global(
                        dia_pipeline, train_data["dia_X"], DIABETES_FEATURES,
                        "Mean |SHAP| — Diabetes Feature Importance (XGBoost)"
                    )
                    if fig2:
                        st.pyplot(fig2, use_container_width=True)

            with tab_lime:
                st.markdown("**LIME Explanation**")
                if "dia_X" in train_data:
                    fig = lime_explanation(
                        dia_pipeline, train_data["dia_X"], input_values_dia,
                        DIABETES_FEATURES, ["No Diabetes", "Diabetes"],
                        f"LIME — Diabetes Patient (prob={prob:.3f})"
                    )
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Training data not found.")

            with tab_stab:
                st.markdown(
                    "**LIME Stability** — 10 runs with different random seeds.  \n"
                    "Error bars show how much explanations vary due to sampling."
                )
                if "dia_X" in train_data:
                    with st.spinner("Running 10 LIME iterations…"):
                        fig = lime_stability(
                            dia_pipeline, train_data["dia_X"], input_values_dia,
                            DIABETES_FEATURES, ["No Diabetes", "Diabetes"]
                        )
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Training data not found.")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Comparison</h1>
        <p>Performance summary of all trained models across both diseases</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">How to read this page</div>', unsafe_allow_html=True)
    st.markdown(
        "This page shows performance metrics saved during training (Notebooks 02 & 04).  \n"
        "If charts from those notebooks were saved as PNGs, they are displayed below."
    )

    tab_ckd, tab_dia = st.tabs(["🫘 CKD Models", "🩸 Diabetes Models"])

    # Helper to display saved PNG
    def show_saved(path, caption=""):
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.info(f"Chart not found: `{path}`  \nRun Notebook 02/04 to generate it.")

    with tab_ckd:
        st.markdown("#### ROC Curves — CKD")
        show_saved("outputs/roc_curves.png", "ROC curves for all CKD models")
        st.markdown("#### Confusion Matrices — CKD")
        show_saved("outputs/confusion_matrices.png", "CKD confusion matrices")
        st.markdown("#### Calibration Curves — CKD")
        show_saved("outputs/calibration_curves.png", "CKD calibration curves")
        st.markdown("#### Feature Importance — CKD")
        show_saved("outputs/feature_importances.png", "CKD feature importance (RF & XGB)")
        st.markdown("#### SHAP Global — CKD")
        show_saved("outputs/shap/ckd/ckd_shap_beeswarm.png",    "CKD SHAP beeswarm")
        show_saved("outputs/shap/ckd/ckd_shap_bar_importance.png", "CKD SHAP bar")

    with tab_dia:
        st.markdown("#### ROC Curves — Diabetes")
        show_saved("outputs/roc_curves_diabetes.png", "ROC curves for all Diabetes models")
        st.markdown("#### Confusion Matrices — Diabetes")
        show_saved("outputs/confusion_matrices_diabetes.png")
        st.markdown("#### Calibration Curves — Diabetes")
        show_saved("outputs/calibration_curves_diabetes.png")
        st.markdown("#### Feature Importance — Diabetes")
        show_saved("outputs/feature_importances_diabetes.png")
        st.markdown("#### SHAP Global — Diabetes")
        show_saved("outputs/shap/diabetes/dia_shap_beeswarm.png")
        show_saved("outputs/shap/diabetes/dia_shap_bar_importance.png")

    # ── SHAP vs LIME comparison images ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">SHAP vs LIME Comparison</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        show_saved("outputs/comparison/ckd_shap_vs_lime.png",  "CKD: SHAP vs LIME")
    with c2:
        show_saved("outputs/comparison/dia_shap_vs_lime.png", "Diabetes: SHAP vs LIME")

    # ── Methods comparison table ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">SHAP vs LIME: Method Comparison</div>', unsafe_allow_html=True)
    comparison_df = pd.DataFrame({
        "Property":        ["Basis", "Scope", "Consistency", "Model-agnostic",
                            "Speed", "Stability", "Use in this project"],
        "SHAP":            ["Shapley values (game theory)", "Global + local",
                            "Exact & consistent", "Yes (TreeExplainer for trees)",
                            "Fast (TreeExplainer)", "Always same result",
                            "Global importance + per-patient waterfall"],
        "LIME":            ["Local linear surrogate", "Local only",
                            "Approximate", "Yes (any classifier)",
                            "Moderate (sampling)", "Varies per run — stability check needed",
                            "Per-patient bars + 10-run stability check"],
    })
    st.dataframe(comparison_df.set_index("Property"), use_container_width=True)