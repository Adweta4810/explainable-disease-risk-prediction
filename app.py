import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── src imports ───────────────────────────────────────────────────────────────
from src.model_training   import load_models
from src.evaluation       import compute_metrics
from src.data_preprocessing import load_cleaned_ckd, load_cleaned_diabetes
from src.shap_explainer   import SHAPExplainer
from src.lime_explainer   import LIMEExplainer

st.set_page_config(page_title="PredictCare", page_icon="💚", layout="wide")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── Paths ─────────────────────────────────────────────────────────────────────
CKD_CLEANED_PATH      = os.path.join(DATA_DIR, "Chronic_Kidney_Disease", "ckd_cleaned.csv")
DIABETES_CLEANED_PATH = os.path.join(DATA_DIR, "Diabetes", "diabetes_cleaned.csv")

DISCLAIMER = """
⚠️ This system is only for educational and research purposes.
It is not a medical diagnosis tool. Please consult a doctor for medical decisions.
"""

# ── Feature definitions ───────────────────────────────────────────────────────
CKD_FEATURES = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba",
    "bgr","bu","sc","sod","pot","hemo","pcv","wc","rc",
    "htn","dm","cad","appet","pe","ane",
]
CKD_BINARY_FEATURES = {
    "rbc":  {"normal": 0, "abnormal": 1},
    "pc":   {"normal": 0, "abnormal": 1},
    "pcc":  {"notpresent": 0, "present": 1},
    "ba":   {"notpresent": 0, "present": 1},
    "htn":  {"no": 0, "yes": 1},
    "dm":   {"no": 0, "yes": 1},
    "cad":  {"no": 0, "yes": 1},
    "appet":{"good": 1, "poor": 0},
    "pe":   {"no": 0, "yes": 1},
    "ane":  {"no": 0, "yes": 1},
}
CKD_NUMERIC_RANGES = {
    "age": (2.0,90.0,51.0), "bp":(50.0,180.0,76.0), "sg":(1.005,1.025,1.020),
    "al":  (0.0,5.0,0.0),   "su":(0.0,5.0,0.0),     "bgr":(44.0,490.0,121.0),
    "bu":  (1.5,391.0,42.0),"sc":(0.4,76.0,1.1),     "sod":(4.5,163.0,137.0),
    "pot": (2.5,47.0,4.6),  "hemo":(3.1,17.8,12.7),  "pcv":(9.0,54.0,39.0),
    "wc":  (2200.,26400.,8000.), "rc":(2.1,8.0,4.7),
}

DIABETES_FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age",
]
DIABETES_RANGES = {
    "Pregnancies":(0,17,3), "Glucose":(0,199,117), "BloodPressure":(0,122,72),
    "SkinThickness":(0,99,23), "Insulin":(0,846,30), "BMI":(0.0,67.1,32.0),
    "DiabetesPedigreeFunction":(0.078,2.42,0.372), "Age":(21,81,29),
}
# Medians used to replace biologically invalid zero values (mirrors data_preprocessing.py)
DIABETES_ZERO_IMP_MEDIANS = {
    "Glucose":117.0, "BloodPressure":72.0, "SkinThickness":23.0,
    "Insulin":125.0, "BMI":32.0,
}

PALETTE = ["#087331","#2ea85a","#9df071"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@media (prefers-color-scheme: light) {
    .stApp { background: linear-gradient(120deg,#f7fbf8,#ffffff); color:#0f172a; }
    .card  { background: white; border:1px solid #e5e7eb; color:#0f172a;
             box-shadow:0 10px 30px rgba(15,23,42,.07); }
    .hero-subtitle { color:#667085; }
    .kpi-value     { color:#087331; }
}
@media (prefers-color-scheme: dark) {
    .stApp { background: linear-gradient(120deg,#0d1f14,#0a1a0f); color:#e2e8f0; }
    .card  { background: #1a2e20; border:1px solid #2d4a35; color:#e2e8f0;
             box-shadow:0 10px 30px rgba(0,0,0,.3); }
    .hero-subtitle { color:#94a3b8; }
    .kpi-value     { color:#4ade80; }
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#064d25,#033d1d) !important; }
section[data-testid="stSidebar"] * { color:white !important; }
.card { padding:22px; border-radius:22px; margin-bottom:18px; }
.logo { font-size:30px; font-weight:800; margin-bottom:25px; }
.logo span { color:#9df071; }
.hero-title { font-size:34px; font-weight:800; }
.hero-subtitle { font-size:16px; margin-bottom:25px; }
.kpi-value { font-size:40px; font-weight:800; }
.low      { color:#166534; background:#dcfce7; padding:6px 12px; border-radius:999px; font-weight:700; }
.moderate { color:#854d0e; background:#fef9c3; padding:6px 12px; border-radius:999px; font-weight:700; }
.high     { color:#991b1b; background:#fee2e2; padding:6px 12px; border-radius:999px; font-weight:700; }
.stButton button {
    background:#087331 !important; color:white !important;
    border-radius:12px !important; border:none !important; font-weight:700 !important;
}
@media (prefers-color-scheme: dark) {
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background:#1a2e20 !important; color:#e2e8f0 !important; border-color:#2d4a35 !important;
    }
    .stDataFrame { background:#1a2e20 !important; }
    label { color:#e2e8f0 !important; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Cached resource loaders
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_ckd_models():
    """Load all three CKD pipelines via src.model_training.load_models."""
    return load_models(os.path.join(MODEL_DIR, "CKD"), dataset_label="ckd")

@st.cache_resource
def get_diabetes_models():
    """Load all three Diabetes pipelines via src.model_training.load_models."""
    return load_models(os.path.join(MODEL_DIR, "diabetes"), dataset_label="diabetes")

@st.cache_data
def get_ckd_data():
    """Load and split cleaned CKD data via src.data_preprocessing."""
    return load_cleaned_ckd(CKD_CLEANED_PATH)

@st.cache_data
def get_diabetes_data():
    """Load and split cleaned Diabetes data via src.data_preprocessing."""
    return load_cleaned_diabetes(DIABETES_CLEANED_PATH)

@st.cache_data
def get_test_metrics(disease: str):
    """
    Compute live hold-out test metrics for all three models using
    src.evaluation.compute_metrics — so numbers always match the
    currently loaded models (no more hardcoded dicts).
    """
    if disease == "CKD":
        models  = get_ckd_models()
        data    = get_ckd_data()
        cm_labels = ["Not CKD", "CKD"]
    else:
        models  = get_diabetes_models()
        data    = get_diabetes_data()
        cm_labels = ["No Diabetes", "Diabetes"]

    X_test, y_test = data["X_test"], data["y_test"]
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, y_pred, y_prob)
    return results, X_test, y_test, cm_labels

@st.cache_data
def load_shap_roc():
    p = os.path.join(BASE_DIR, "shap_roc_data.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ════════════════════════════════════════════════════════════════════════════

def predict_probability(model, row: pd.DataFrame) -> float:
    proba   = model.predict_proba(row)[0]
    final   = list(model.named_steps.values())[-1]
    classes = list(final.classes_) if hasattr(final, "classes_") else list(range(len(proba)))
    pos_idx = classes.index(1) if 1 in classes else -1
    return float(proba[pos_idx]) * 100

def risk_level(prob: float):
    if prob >= 70:   return "High Risk",     "high"
    elif prob >= 40: return "Moderate Risk", "moderate"
    return "Low Risk", "low"

def encode_ckd_input(input_values: dict) -> dict:
    """Encode raw CKD form values to numeric using CKD_BINARY_FEATURES."""
    encoded = {}
    for feat in CKD_FEATURES:
        val = input_values[feat]
        encoded[feat] = float(CKD_BINARY_FEATURES[feat][val]) if feat in CKD_BINARY_FEATURES else float(val)
    return encoded

def preprocess_diabetes_input(raw_dict: dict) -> pd.DataFrame:
    """
    Replace biologically invalid zeros with medians (mirrors
    data_preprocessing.preprocess_diabetes step 2), then return a
    DataFrame ready for prediction.  No manual scaling is done here —
    the Diabetes pipeline carries its own StandardScaler internally.
    """
    d = dict(raw_dict)
    for col, med in DIABETES_ZERO_IMP_MEDIANS.items():
        if d.get(col, 1) == 0:
            d[col] = med
    return pd.DataFrame([[d[f] for f in DIABETES_FEATURES]], columns=DIABETES_FEATURES)

def get_feature_importance(model, features: list) -> pd.DataFrame:
    """Extract feature importances from the classifier step of the pipeline."""
    final = list(model.named_steps.values())[-1]
    if hasattr(final, "feature_importances_"):
        values = final.feature_importances_
    elif hasattr(final, "coef_"):
        values = np.abs(final.coef_).ravel()
    else:
        values = np.ones(len(features))
    values = values[:len(features)]
    return pd.DataFrame({"Feature": features[:len(values)], "Importance": values}) \
             .sort_values("Importance", ascending=False)

def explain_risk_causes(disease: str, input_values: dict, importance_df: pd.DataFrame) -> list:
    diabetes_rules = {
        "Glucose":                 "High glucose is one of the strongest diabetes risk factors.",
        "BMI":                     "High BMI can increase diabetes risk.",
        "Age":                     "Older age can increase diabetes risk.",
        "Insulin":                 "Abnormal insulin can affect blood sugar control.",
        "BloodPressure":           "High blood pressure can increase diabetes-related risk.",
        "DiabetesPedigreeFunction":"Family history can increase diabetes risk.",
        "Pregnancies":             "Pregnancy count may influence diabetes risk.",
        "SkinThickness":           "Skin thickness may relate to body fat level.",
    }
    ckd_rules = {
        "sc":  "High serum creatinine may indicate reduced kidney function.",
        "bu":  "High blood urea may suggest kidney filtering problems.",
        "al":  "Albumin in urine may indicate kidney damage.",
        "hemo":"Low hemoglobin may be linked with CKD-related anemia.",
        "bp":  "High blood pressure can damage kidney function.",
        "bgr": "High blood glucose can increase kidney disease risk.",
        "htn": "Hypertension history increases CKD risk.",
        "dm":  "Diabetes history increases CKD risk.",
        "sg":  "Abnormal specific gravity may suggest kidney concentration problems.",
        "pcv": "Low packed cell volume may be linked with kidney disease.",
        "ane": "Anemia may be associated with CKD.",
        "pe":  "Pedal edema may be related to kidney problems.",
    }
    rules = diabetes_rules if disease == "Diabetes" else ckd_rules
    return [
        {
            "Feature":       r["Feature"],
            "Patient Value": input_values.get(r["Feature"], "N/A"),
            "Importance":    r["Importance"],
            "Reason":        rules.get(r["Feature"], "This feature strongly influenced the model prediction."),
        }
        for _, r in importance_df.head(5).iterrows()
    ]

def page_header(title: str, subtitle: str):
    st.markdown(
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown('<div class="logo">💚 Predict<span>Care</span></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", [
    "Dashboard / Overview", "Prediction", "Explainability",
    "Model Performance", "Export / Report",
])
st.sidebar.markdown("---")
st.sidebar.info(DISCLAIMER)


# ════════════════════════════════════════════════════════════════════════════
# Dashboard / Overview
# ════════════════════════════════════════════════════════════════════════════

if page == "Dashboard / Overview":
    page_header("Dashboard Overview", "AI-powered disease risk prediction for CKD and Diabetes.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="card"><h4>CKD Records</h4><div class="kpi-value">400</div><span class="low">Dataset Ready</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>Diabetes Records</h4><div class="kpi-value">768</div><span class="low">Dataset Ready</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>Model Status</h4><div class="kpi-value">Ready</div><span class="low">6 Models</span></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><h4>Total Features</h4><div class="kpi-value">{len(CKD_FEATURES)+len(DIABETES_FEATURES)}</div><span class="moderate">Clinical Inputs</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><b>Dataset Summary</b><br>This dashboard supports CKD and Diabetes prediction using machine learning pipelines.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Prediction
# ════════════════════════════════════════════════════════════════════════════

elif page == "Prediction":
    page_header("Prediction Page", "Enter patient values and predict disease risk.")

    disease    = st.selectbox("Select Disease", ["CKD", "Diabetes"])
    model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    # Load the selected model from src
    all_models = get_ckd_models() if disease == "CKD" else get_diabetes_models()
    model      = all_models[model_name]
    features   = CKD_FEATURES if disease == "CKD" else DIABETES_FEATURES

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient Input Form")
    input_values = {}

    cols = st.columns(3)
    if disease == "CKD":
        for i, feat in enumerate(features):
            with cols[i % 3]:
                if feat in CKD_BINARY_FEATURES:
                    options = list(CKD_BINARY_FEATURES[feat].keys())
                    input_values[feat] = st.selectbox(feat, options=options, key=f"ckd_{feat}")
                else:
                    lo, hi, med = CKD_NUMERIC_RANGES[feat]
                    input_values[feat] = st.number_input(feat, value=med, min_value=lo, max_value=hi, key=f"ckd_{feat}")
    else:
        for i, feat in enumerate(features):
            with cols[i % 3]:
                lo, hi, med = DIABETES_RANGES[feat]
                input_values[feat] = st.number_input(feat, value=float(med), min_value=float(lo), max_value=float(hi), key=f"dia_{feat}")

    predict_btn = st.button("Predict Risk")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        # Build model-ready row using src helpers
        if disease == "CKD":
            encoded = encode_ckd_input(input_values)
            row = pd.DataFrame([encoded], columns=features)
        else:
            row = preprocess_diabetes_input(input_values)

        probability      = predict_probability(model, row)
        level, css_class = risk_level(probability)

        st.session_state["latest_result"] = {
            "Disease": disease, "Model": model_name,
            "Risk Percentage": round(probability, 2), "Risk Level": level,
            **input_values,
        }

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="card"><h2>Prediction Result</h2><div class="kpi-value">{probability:.2f}%</div><span class="{css_class}">{level}</span></div>', unsafe_allow_html=True)
        with col2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=probability,
                title={"text": "Risk Percentage"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#087331"},
                       "steps": [{"range": [0, 40], "color": "#dcfce7"},
                                  {"range": [40, 70], "color": "#fef3c7"},
                                  {"range": [70, 100], "color": "#fee2e2"}]}))
            gauge.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge, key="pred_gauge")

        importance_df = get_feature_importance(model, features).head(5)
        risk_causes   = explain_risk_causes(disease, input_values, importance_df)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What Causes This Risk?")
        for item in risk_causes:
            st.markdown(f"### {item['Feature']}")
            st.write(f"**Patient value:** {item['Patient Value']}")
            st.write(f"**Model importance:** {round(float(item['Importance']), 4)}")
            st.warning(item["Reason"])

        if level == "High Risk":
            st.error("The model predicted High Risk because important indicators strongly influenced the disease class.")
        elif level == "Moderate Risk":
            st.warning("The model predicted Moderate Risk because some important indicators may need attention.")
        else:
            st.success("The model predicted Low Risk because the important indicators appear closer to safer ranges.")
        st.info(DISCLAIMER)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Explainability
# ════════════════════════════════════════════════════════════════════════════

elif page == "Explainability":
    page_header("Explainability", "SHAP-powered global and per-patient explanations.")

    disease  = st.selectbox("Select Disease", ["CKD", "Diabetes"], key="exp_disease")
    shap_roc = load_shap_roc()
    key      = "ckd" if disease == "CKD" else "diabetes"

    # ── Global SHAP bar ───────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    model_label = "Random Forest" if disease == "CKD" else "XGBoost"
    st.subheader(f"🌍 Global SHAP Feature Importance  ({model_label})")
    st.caption("Mean |SHAP value| across all patients — higher = more influential on prediction")

    if shap_roc:
        shap_data  = shap_roc[key]["shap"]
        feat_names = shap_data["features"]
        mean_abs   = shap_data["mean_abs"]
        imp_df = pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_abs})
        imp_df = imp_df.sort_values("Mean |SHAP|", ascending=True).tail(12)
        bar_shap = go.Figure(go.Bar(
            x=imp_df["Mean |SHAP|"], y=imp_df["Feature"], orientation="h",
            marker=dict(color=imp_df["Mean |SHAP|"],
                        colorscale=[[0,"#dcfce7"],[0.5,"#2ea85a"],[1,"#064d25"]],showscale=False),
            text=[f"{v:.4f}" for v in imp_df["Mean |SHAP|"]], textposition="outside"))
        bar_shap.update_layout(height=420, margin=dict(t=10,b=10,l=10,r=60),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               xaxis_title="Mean |SHAP Value|")
        st.plotly_chart(bar_shap, key="shap_global_bar")
    else:
        st.warning("shap_roc_data.json not found. Place it in the same folder as app.py.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    if shap_roc:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🐝 SHAP Beeswarm Plot")
        st.caption("Each dot = one patient. Red = high feature value, Blue = low.")
        sv    = np.array(shap_data["shap_values"])
        fv    = np.array(shap_data["feature_values"])
        order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:10]
        bee_fig = go.Figure()
        rng = np.random.default_rng(42)
        for rank, fi in enumerate(reversed(order)):
            fname  = feat_names[fi]
            x_vals = sv[:, fi]
            fv_col = fv[:, fi]
            fv_norm = (fv_col - fv_col.min()) / ((fv_col.max() - fv_col.min()) + 1e-9)
            y_jit   = rank + rng.uniform(-0.3, 0.3, size=len(x_vals))
            colors_bee = [f"rgb({int(255*v)},{int(50+80*(1-v))},{int(255*(1-v))})" for v in fv_norm]
            bee_fig.add_trace(go.Scatter(
                x=x_vals, y=y_jit, mode="markers",
                marker=dict(size=4, color=colors_bee, opacity=0.6),
                name=fname, showlegend=False,
                hovertemplate=f"<b>{fname}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{customdata:.3f}}",
                customdata=fv_col))
        feat_labels = [feat_names[fi] for fi in reversed(order)]
        bee_fig.update_layout(
            height=500, xaxis_title="SHAP Value (impact on model output)",
            yaxis=dict(tickvals=list(range(len(feat_labels))), ticktext=feat_labels),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10,b=10,l=10,r=10),
            shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(feat_labels)-0.5,
                         line=dict(color="#aaa", dash="dash", width=1))])
        st.plotly_chart(bee_fig, key="shap_beeswarm")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-patient SHAP waterfall (via SHAPExplainer) ────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔬 Per-Patient SHAP Explanation")

    if "latest_result" in st.session_state and st.session_state["latest_result"]["Disease"] == disease:
        result      = st.session_state["latest_result"]
        prob        = result["Risk Percentage"]
        level_label = result["Risk Level"]
        st.success(f"Using last prediction — {level_label} ({prob:.2f}%)")

        try:
            if disease == "CKD":
                encoded    = encode_ckd_input({f: result.get(f, 0) for f in CKD_FEATURES})
                patient_df = pd.DataFrame([encoded], columns=CKD_FEATURES)
                feat_list  = CKD_FEATURES
                m = get_ckd_models()["Random Forest"]
                # SHAPExplainer: no scaler needed (RF pipeline has no scaler step)
                shap_exp = SHAPExplainer(
                    model_pipeline=m, X_test=patient_df,
                    feature_names=feat_list,
                    class_names=["Not CKD", "CKD"],
                    output_dir=os.path.join(BASE_DIR, "outputs", "shap"),
                    dataset_label="CKD",
                    scale_for_shap=False,
                )
            else:
                patient_df = preprocess_diabetes_input({f: result.get(f, 0) for f in DIABETES_FEATURES})
                feat_list  = DIABETES_FEATURES
                m = get_diabetes_models()["XGBoost"]
                # XGBoost pipeline has a scaler step — pass scale_for_shap=True
                shap_exp = SHAPExplainer(
                    model_pipeline=m, X_test=patient_df,
                    feature_names=feat_list,
                    class_names=["No Diabetes", "Diabetes"],
                    output_dir=os.path.join(BASE_DIR, "outputs", "shap"),
                    dataset_label="Diabetes",
                    scale_for_shap=True,
                )

            shap_exp.compute_shap_values()
            sv_patient = shap_exp.shap_values_pos[0]
            base_val   = shap_exp.expected_value_pos

            contrib_df = pd.DataFrame({
                "Feature": feat_list,
                "Value":   patient_df.values[0],
                "SHAP":    sv_patient,
            })
            contrib_df = contrib_df.reindex(
                contrib_df["SHAP"].abs().sort_values(ascending=False).index
            ).head(10).sort_values("SHAP")

            colors_wf = ["#b91c1c" if v > 0 else "#166534" for v in contrib_df["SHAP"]]
            wf_fig = go.Figure(go.Bar(
                x=contrib_df["SHAP"],
                y=[f"{r['Feature']} = {r['Value']:.2f}" for _, r in contrib_df.iterrows()],
                orientation="h", marker_color=colors_wf,
                text=[f"{v:+.4f}" for v in contrib_df["SHAP"]], textposition="outside"))
            wf_fig.update_layout(
                height=420,
                title=f"Base value: {base_val:.3f}  →  Prediction: {prob/100:.3f}",
                xaxis_title="SHAP contribution (red=toward disease, green=away)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40,b=10,l=10,r=60),
                shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(contrib_df)-0.5,
                             line=dict(color="#aaa", dash="dash", width=1))])
            st.plotly_chart(wf_fig, key="shap_waterfall")

        except Exception as e:
            st.error(f"Could not compute SHAP: {e}")
    else:
        st.info("💡 Run a prediction first on the Prediction page — then come back here for a patient-specific SHAP explanation.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-patient LIME (via LIMEExplainer) ─────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🍋 LIME Per-Patient Explanation")
    st.caption("LIME fits a simple local model around this patient to explain the prediction.")

    if "latest_result" in st.session_state and st.session_state["latest_result"]["Disease"] == disease:
        result      = st.session_state["latest_result"]
        prob        = result["Risk Percentage"]
        level_label = result["Risk Level"]
        st.success(f"Using last prediction — {level_label} ({prob:.2f}%)")

        try:
            if disease == "CKD":
                data      = get_ckd_data()
                X_train_l = data["X_train"]
                feat_list = CKD_FEATURES
                cls_names = ["Not CKD", "CKD"]
                m         = get_ckd_models()["Random Forest"]
                encoded   = encode_ckd_input({f: result.get(f, 0) for f in CKD_FEATURES})
                patient_df = pd.DataFrame([encoded], columns=feat_list)
            else:
                data      = get_diabetes_data()
                X_train_l = data["X_train"]
                feat_list = DIABETES_FEATURES
                cls_names = ["No Diabetes", "Diabetes"]
                m         = get_diabetes_models()["XGBoost"]
                patient_df = preprocess_diabetes_input({f: result.get(f, 0) for f in DIABETES_FEATURES})

            # Build LIMEExplainer via src — X_test is just the one patient
            lime_exp = LIMEExplainer(
                model_pipeline=m,
                X_train=X_train_l,
                X_test=patient_df,
                feature_names=feat_list,
                class_names=cls_names,
                output_dir=os.path.join(BASE_DIR, "outputs", "lime"),
                dataset_label=disease,
            )
            lime_table = lime_exp.explain_patient(
                patient_idx=0, save_plot=False, save_csv=False, show_plot=False
            )

            lime_df = lime_table.sort_values("Contribution")
            colors_lime = ["#b91c1c" if v > 0 else "#166534" for v in lime_df["Contribution"]]
            lime_fig = go.Figure(go.Bar(
                x=lime_df["Contribution"],
                y=lime_df["Feature condition"],
                orientation="h", marker_color=colors_lime,
                text=[f"{v:+.4f}" for v in lime_df["Contribution"]],
                textposition="outside"))
            lime_fig.update_layout(
                height=420,
                title=f"LIME Local Explanation — {level_label} ({prob:.2f}%)",
                xaxis_title="Contribution (red = toward disease, green = away from disease)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40,b=10,l=10,r=60),
                shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(lime_df)-0.5,
                             line=dict(color="#aaa", dash="dash", width=1))])
            st.plotly_chart(lime_fig, key="lime_waterfall")

            # SHAP vs LIME comparison table
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("⚖️ SHAP vs LIME — Key Differences")
            comp_df = pd.DataFrame({
                "":         ["Approach", "Scope", "Stability", "Best for"],
                "SHAP 🔵":  ["Game-theory (Shapley values)", "Global + per-patient", "High — consistent results", "Understanding overall model behaviour"],
                "LIME 🟡":  ["Local linear approximation", "Per-patient only", "Can vary between runs", "Quick local explanation per patient"],
            })
            st.table(comp_df.set_index(""))
            st.markdown('</div>', unsafe_allow_html=True)

        except FileNotFoundError:
            st.warning("Training data CSV not found. LIME needs the cleaned dataset files in the data/ folder.")
        except Exception as e:
            st.error(f"Could not compute LIME explanation: {e}")
    else:
        st.info("💡 Run a prediction first on the Prediction page — then come back here for a LIME explanation.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.info(DISCLAIMER)


# ════════════════════════════════════════════════════════════════════════════
# Model Performance  — live metrics via src.evaluation.compute_metrics
# ════════════════════════════════════════════════════════════════════════════

elif page == "Model Performance":
    page_header("Model Performance", "Real test-set results from trained pipelines.")

    disease_tab = st.selectbox("Select Disease", ["CKD", "Diabetes"], key="perf_disease")

    # All metrics computed live — no hardcoded dicts
    metrics_raw, X_test, y_test, cm_labels = get_test_metrics(disease_tab)

    # Reformat keys to match display names (compute_metrics uses "ROC-AUC" etc.)
    metrics_dict = {
        name: {k.replace("-", "_"): v for k, v in m.items()}
        for name, m in metrics_raw.items()
    }

    best_model = max(metrics_dict, key=lambda m: metrics_dict[m]["ROC_AUC"])
    bm = metrics_dict[best_model]
    st.markdown(f"<div class='card'><b>🏆 Best Model (by ROC-AUC): {best_model}</b></div>", unsafe_allow_html=True)

    k_cols = st.columns(5)
    for col, label, km in zip(k_cols,
        ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
        ["Accuracy","Precision","Recall","F1","ROC_AUC"]):
        val = bm[km]
        css = "low" if val >= 0.85 else ("moderate" if val >= 0.70 else "high")
        col.markdown(
            f'<div class="card" style="text-align:center">'
            f'<div style="font-size:13px;color:#667085">{label}</div>'
            f'<div class="kpi-value" style="font-size:28px">{val:.4f}</div>'
            f'<span class="{css}">{"Excellent" if val>=0.85 else ("Good" if val>=0.70 else "Fair")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Full metrics table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Full Metrics Comparison")
    rows = [
        {"Model": mn, "Accuracy": f"{m['Accuracy']:.4f}", "Precision": f"{m['Precision']:.4f}",
         "Recall": f"{m['Recall']:.4f}", "F1-Score": f"{m['F1']:.4f}",
         "ROC-AUC": f"{m['ROC_AUC']:.4f}", "Kappa": f"{m['Kappa']:.4f}", "Brier ↓": f"{m['Brier']:.4f}"}
        for mn, m in metrics_dict.items()
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bar chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Metric Bar Chart")
    metric_choice = st.selectbox("Choose metric", ["Accuracy","Precision","Recall","F1","ROC_AUC","Kappa"], key="metric_bar")
    bar_fig = go.Figure()
    for i, (mn, m) in enumerate(metrics_dict.items()):
        bar_fig.add_trace(go.Bar(
            name=mn, x=[mn], y=[m[metric_choice]],
            marker_color=PALETTE[i],
            text=[f"{m[metric_choice]:.4f}"], textposition="outside"))
    bar_fig.update_layout(
        yaxis=dict(range=[0, 1.1], title=metric_choice),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, height=350, margin=dict(t=20,b=20))
    st.plotly_chart(bar_fig, key="perf_bar_chart")
    st.markdown('</div>', unsafe_allow_html=True)

    # ROC curves
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📉 ROC Curves")
    shap_roc = load_shap_roc()
    if shap_roc:
        roc_key = "ckd" if disease_tab == "CKD" else "diabetes"
        roc_fig = go.Figure()
        for (mn_roc, rdata), color in zip(shap_roc[roc_key]["roc"].items(), PALETTE):
            roc_fig.add_trace(go.Scatter(
                x=rdata["fpr"], y=rdata["tpr"], mode="lines",
                name=f"{mn_roc} (AUC={rdata['auc']:.4f})",
                line=dict(color=color, width=2.5)))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#aaa", dash="dash", width=1), name="Random Classifier"))
        roc_fig.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.55, y=0.1), margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(roc_fig, key="perf_roc_curves")
    else:
        st.warning("shap_roc_data.json not found next to app.py.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Confusion matrices — computed live from current models + test data
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔢 Confusion Matrices")
    all_models = get_ckd_models() if disease_tab == "CKD" else get_diabetes_models()
    cm_cols = st.columns(3)
    for idx, (col, (mn_cm, model)) in enumerate(zip(cm_cols, all_models.items())):
        with col:
            from sklearn.metrics import confusion_matrix as _cm
            y_pred_cm = model.predict(X_test)
            tn, fp, fn, tp = _cm(y_test, y_pred_cm).ravel()
            st.markdown(f"**{mn_cm}**")
            heatmap = go.Figure(go.Heatmap(
                z=[[tn, fp],[fn, tp]],
                x=[f"Pred {cm_labels[0]}", f"Pred {cm_labels[1]}"],
                y=[f"True {cm_labels[0]}", f"True {cm_labels[1]}"],
                colorscale=[[0,"#f0fdf4"],[1,"#087331"]],
                text=[[str(tn), str(fp)],[str(fn), str(tp)]],
                texttemplate="%{text}", textfont={"size":18,"color":"black"}, showscale=False))
            heatmap.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10),
                                  paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(heatmap, key=f"cm_{disease_tab}_{idx}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Overfitting check — CV AUC from shap_roc_data.json vs live test AUC
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 CV vs Hold-out AUC (Overfitting Check)")
    if shap_roc:
        roc_key = "ckd" if disease_tab == "CKD" else "diabetes"
        ov_rows = []
        for mn, m in metrics_dict.items():
            test_auc = m["ROC_AUC"]
            cv_auc   = shap_roc[roc_key]["roc"].get(mn, {}).get("cv_auc", None)
            if cv_auc is not None:
                gap    = cv_auc - test_auc
                status = "⚠ Overfit" if gap > 0.05 else "✅ OK"
                ov_rows.append({"Model": mn, "CV AUC": round(cv_auc,4),
                                "Test AUC": round(test_auc,4), "Gap": f"{gap:+.4f}", "Status": status})
        if ov_rows:
            st.dataframe(pd.DataFrame(ov_rows).set_index("Model"), use_container_width=True)
        else:
            st.info("Add 'cv_auc' keys to shap_roc_data.json under each model's roc entry to enable this check.")
    else:
        st.warning("shap_roc_data.json not found.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.info(DISCLAIMER)


# ════════════════════════════════════════════════════════════════════════════
# Export / Report
# ════════════════════════════════════════════════════════════════════════════

elif page == "Export / Report":
    page_header("Export / Report", "Download latest prediction result.")
    if "latest_result" in st.session_state:
        result_df = pd.DataFrame([st.session_state["latest_result"]])
        st.dataframe(result_df, use_container_width=True)
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "predictcare_result.csv", "text/csv")
    else:
        st.warning("No prediction result found yet. Go to Prediction page first.")
    st.error(DISCLAIMER)