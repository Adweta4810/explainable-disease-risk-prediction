import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="PredictCare", page_icon="💚", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CKD_MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODEL_DIR, "CKD", "ckd_lr_pipeline.pkl"),
    "Random Forest":       os.path.join(MODEL_DIR, "CKD", "ckd_rf_pipeline.pkl"),
    "XGBoost":             os.path.join(MODEL_DIR, "CKD", "ckd_xgb_pipeline.pkl"),
}
DIABETES_MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODEL_DIR, "diabetes", "diabetes_lr_pipeline.pkl"),
    "Random Forest":       os.path.join(MODEL_DIR, "diabetes", "diabetes_rf_pipeline.pkl"),
    "XGBoost":             os.path.join(MODEL_DIR, "diabetes", "diabetes_xgb_pipeline.pkl"),
}

DISCLAIMER = """
\u26a0\ufe0f This system is only for educational and research purposes.
It is not a medical diagnosis tool. Please consult a doctor for medical decisions.
"""

CKD_FEATURES = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba",
    "bgr","bu","sc","sod","pot","hemo","pcv","wc","rc",
    "htn","dm","cad","appet","pe","ane",
]
CKD_BINARY_FEATURES = {
    "rbc":  {"normal":0,"abnormal":1},
    "pc":   {"normal":0,"abnormal":1},
    "pcc":  {"notpresent":0,"present":1},
    "ba":   {"notpresent":0,"present":1},
    "htn":  {"no":0,"yes":1},
    "dm":   {"no":0,"yes":1},
    "cad":  {"no":0,"yes":1},
    "appet":{"good":1,"poor":0},
    "pe":   {"no":0,"yes":1},
    "ane":  {"no":0,"yes":1},
}
CKD_NUMERIC_RANGES = {
    "age": (2.0,90.0,51.0),"bp":(50.0,180.0,76.0),"sg":(1.005,1.025,1.020),
    "al":(0.0,5.0,0.0),"su":(0.0,5.0,0.0),"bgr":(44.0,490.0,121.0),
    "bu":(1.5,391.0,42.0),"sc":(0.4,76.0,1.1),"sod":(4.5,163.0,137.0),
    "pot":(2.5,47.0,4.6),"hemo":(3.1,17.8,12.7),"pcv":(9.0,54.0,39.0),
    "wc":(2200.,26400.,8000.),"rc":(2.1,8.0,4.7),
}

DIABETES_FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
DIABETES_RANGES = {
    "Pregnancies":(0,17,3),"Glucose":(0,199,117),"BloodPressure":(0,122,72),
    "SkinThickness":(0,99,23),"Insulin":(0,846,30),"BMI":(0.0,67.1,32.0),
    "DiabetesPedigreeFunction":(0.078,2.42,0.372),"Age":(21,81,29),
}
DIABETES_SCALER_MEAN  = [3.8451,121.6562,72.3867,29.1081,140.6719,32.4552,0.4719,33.2409]
DIABETES_SCALER_SCALE = [3.3674, 30.4185,12.0888, 8.7855, 86.3268, 6.8707,0.3311,11.7526]
DIABETES_ZERO_IMP_MEDIANS = {
    "Glucose":117.0,"BloodPressure":72.0,"SkinThickness":23.0,"Insulin":125.0,"BMI":32.0,
}

st.markdown("""
<style>
.stApp{background:linear-gradient(120deg,#f7fbf8,#ffffff);color:#0f172a}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#064d25,#033d1d)}
section[data-testid="stSidebar"] *{color:white !important}
.card{background:white;padding:22px;border-radius:22px;border:1px solid #e5e7eb;box-shadow:0 10px 30px rgba(15,23,42,.07);margin-bottom:18px}
.logo{font-size:30px;font-weight:800;margin-bottom:25px}
.logo span{color:#9df071}
.hero-title{font-size:34px;font-weight:800}
.hero-subtitle{color:#667085;font-size:16px;margin-bottom:25px}
.kpi-value{font-size:40px;font-weight:800;color:#087331}
.low{color:#166534;background:#dcfce7;padding:6px 12px;border-radius:999px;font-weight:700}
.moderate{color:#b45309;background:#fef3c7;padding:6px 12px;border-radius:999px;font-weight:700}
.high{color:#b91c1c;background:#fee2e2;padding:6px 12px;border-radius:999px;font-weight:700}
.stButton button{background:#087331 !important;color:white !important;border-radius:12px !important;border:none !important;font-weight:700 !important}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    return joblib.load(path)

@st.cache_data
def load_shap_roc():
    p = os.path.join(BASE_DIR, "shap_roc_data.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def predict_probability(model, row):
    if not hasattr(model, "predict_proba"):
        return float(model.predict(row)[0]) * 100
    proba   = model.predict_proba(row)[0]
    final   = list(model.named_steps.values())[-1] if hasattr(model,"named_steps") else model
    classes = list(final.classes_) if hasattr(final,"classes_") else list(range(len(proba)))
    pos_idx = classes.index(1) if 1 in classes else -1
    return float(proba[pos_idx]) * 100

def risk_level(prob):
    if prob >= 70:   return "High Risk",     "high"
    elif prob >= 40: return "Moderate Risk",  "moderate"
    return "Low Risk", "low"

def preprocess_diabetes_input(raw_dict):
    d = dict(raw_dict)
    for col, med in DIABETES_ZERO_IMP_MEDIANS.items():
        if d.get(col, 1) == 0:
            d[col] = med
    arr = np.array([[d[f] for f in DIABETES_FEATURES]], dtype=float)
    arr = (arr - DIABETES_SCALER_MEAN) / DIABETES_SCALER_SCALE
    return pd.DataFrame(arr, columns=DIABETES_FEATURES)

def get_feature_importance(model, features):
    final = list(model.named_steps.values())[-1] if hasattr(model,"named_steps") else model
    if hasattr(final,"feature_importances_"):   values = final.feature_importances_
    elif hasattr(final,"coef_"):                values = np.abs(final.coef_).ravel()
    else:                                       values = np.ones(len(features))
    values = values[:len(features)]
    return pd.DataFrame({"Feature":features[:len(values)],"Importance":values}).sort_values("Importance",ascending=False)

def explain_risk_causes(disease, input_values, importance_df):
    diabetes_rules = {
        "Glucose":"High glucose is one of the strongest diabetes risk factors.",
        "BMI":"High BMI can increase diabetes risk.",
        "Age":"Older age can increase diabetes risk.",
        "Insulin":"Abnormal insulin can affect blood sugar control.",
        "BloodPressure":"High blood pressure can increase diabetes-related risk.",
        "DiabetesPedigreeFunction":"Family history can increase diabetes risk.",
        "Pregnancies":"Pregnancy count may influence diabetes risk.",
        "SkinThickness":"Skin thickness may relate to body fat level.",
    }
    ckd_rules = {
        "sc":"High serum creatinine may indicate reduced kidney function.",
        "bu":"High blood urea may suggest kidney filtering problems.",
        "al":"Albumin in urine may indicate kidney damage.",
        "hemo":"Low hemoglobin may be linked with CKD-related anemia.",
        "bp":"High blood pressure can damage kidney function.",
        "bgr":"High blood glucose can increase kidney disease risk.",
        "htn":"Hypertension history increases CKD risk.",
        "dm":"Diabetes history increases CKD risk.",
        "sg":"Abnormal specific gravity may suggest kidney concentration problems.",
        "pcv":"Low packed cell volume may be linked with kidney disease.",
        "ane":"Anemia may be associated with CKD.",
        "pe":"Pedal edema may be related to kidney problems.",
    }
    rules = diabetes_rules if disease=="Diabetes" else ckd_rules
    return [{"Feature":r["Feature"],"Patient Value":input_values.get(r["Feature"],"N/A"),
             "Importance":r["Importance"],"Reason":rules.get(r["Feature"],"This feature strongly influenced the model prediction.")}
            for _,r in importance_df.head(5).iterrows()]

def page_header(title, subtitle):
    st.markdown(f'<div class="hero-title">{title}</div><div class="hero-subtitle">{subtitle}</div>',
                unsafe_allow_html=True)

PALETTE = ["#087331","#2ea85a","#9df071"]

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="logo">💚 Predict<span>Care</span></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation",
    ["Dashboard / Overview","Prediction","Explainability","Model Performance","Export / Report"])
st.sidebar.markdown("---")
st.sidebar.info(DISCLAIMER)


# ════════════════════════════════════════════════════════════════════════════
if page == "Dashboard / Overview":
    page_header("Dashboard Overview","AI-powered disease risk prediction for CKD and Diabetes.")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="card"><h4>CKD Records</h4><div class="kpi-value">400</div><span class="low">Dataset Ready</span></div>',unsafe_allow_html=True)
    with c2: st.markdown('<div class="card"><h4>Diabetes Records</h4><div class="kpi-value">768</div><span class="low">Dataset Ready</span></div>',unsafe_allow_html=True)
    with c3: st.markdown('<div class="card"><h4>Model Status</h4><div class="kpi-value">Ready</div><span class="low">6 Models</span></div>',unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="card"><h4>Total Features</h4><div class="kpi-value">{len(CKD_FEATURES)+len(DIABETES_FEATURES)}</div><span class="moderate">Clinical Inputs</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><b>Dataset Summary</b><br>This dashboard supports CKD and Diabetes prediction using machine learning pipelines.</div>',unsafe_allow_html=True)


elif page == "Prediction":
    page_header("Prediction Page","Enter patient values and predict disease risk.")
    disease    = st.selectbox("Select Disease",["CKD","Diabetes"])
    model_name = st.selectbox("Select Model",["Logistic Regression","Random Forest","XGBoost"])
    model_path = CKD_MODEL_PATHS[model_name] if disease=="CKD" else DIABETES_MODEL_PATHS[model_name]
    model      = load_model(model_path)

    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("Patient Input Form")
    input_values  = {}
    encoded_values = {}

    if disease == "CKD":
        features = CKD_FEATURES
        cols = st.columns(3)
        for i,feat in enumerate(features):
            with cols[i%3]:
                if feat in CKD_BINARY_FEATURES:
                    options = list(CKD_BINARY_FEATURES[feat].keys())
                    chosen  = st.selectbox(feat, options=options, key=f"ckd_{feat}")
                    input_values[feat]   = chosen
                    encoded_values[feat] = float(CKD_BINARY_FEATURES[feat][chosen])
                else:
                    lo,hi,med = CKD_NUMERIC_RANGES[feat]
                    val = st.number_input(feat,value=med,min_value=lo,max_value=hi,key=f"ckd_{feat}")
                    input_values[feat]   = val
                    encoded_values[feat] = float(val)
    else:
        features = DIABETES_FEATURES
        cols = st.columns(3)
        for i,feat in enumerate(features):
            with cols[i%3]:
                lo,hi,med = DIABETES_RANGES[feat]
                val = st.number_input(feat,value=float(med),min_value=float(lo),max_value=float(hi),key=f"dia_{feat}")
                input_values[feat]   = val
                encoded_values[feat] = float(val)

    predict_btn = st.button("Predict Risk")
    st.markdown('</div>',unsafe_allow_html=True)

    if predict_btn:
        row = pd.DataFrame([encoded_values],columns=features) if disease=="CKD" else preprocess_diabetes_input(encoded_values)
        probability      = predict_probability(model, row)
        level, css_class = risk_level(probability)

        st.session_state["latest_result"] = {
            "Disease":disease,"Model":model_name,
            "Risk Percentage":round(probability,2),"Risk Level":level,
            **input_values,
        }

        col1,col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="card"><h2>Prediction Result</h2><div class="kpi-value">{probability:.2f}%</div><span class="{css_class}">{level}</span></div>',unsafe_allow_html=True)
        with col2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=probability,
                title={"text":"Risk Percentage"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#087331"},
                       "steps":[{"range":[0,40],"color":"#dcfce7"},{"range":[40,70],"color":"#fef3c7"},{"range":[70,100],"color":"#fee2e2"}]}))
            gauge.update_layout(height=320)
            st.plotly_chart(gauge, key="pred_gauge")

        importance_df = get_feature_importance(model, features).head(5)
        risk_causes   = explain_risk_causes(disease, input_values, importance_df)
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("What Causes This Risk?")
        for item in risk_causes:
            st.markdown(f"### {item['Feature']}")
            st.write(f"**Patient value:** {item['Patient Value']}")
            st.write(f"**Model importance:** {round(float(item['Importance']),4)}")
            st.warning(item["Reason"])
        if level=="High Risk":      st.error("The model predicted High Risk because important indicators strongly influenced the disease class.")
        elif level=="Moderate Risk": st.warning("The model predicted Moderate Risk because some important indicators may need attention.")
        else:                        st.success("The model predicted Low Risk because the important indicators appear closer to safer ranges.")
        st.info(DISCLAIMER)
        st.markdown('</div>',unsafe_allow_html=True)


elif page == "Explainability":
    page_header("Explainability","SHAP-powered global and per-patient explanations.")
    disease  = st.selectbox("Select Disease",["CKD","Diabetes"],key="exp_disease")
    shap_roc = load_shap_roc()
    key      = "ckd" if disease=="CKD" else "diabetes"

    # Global SHAP bar
    st.markdown('<div class="card">',unsafe_allow_html=True)
    model_label = "Random Forest" if disease=="CKD" else "XGBoost"
    st.subheader(f"🌍 Global SHAP Feature Importance  ({model_label})")
    st.caption("Mean |SHAP value| across all patients — higher = more influential on prediction")
    if shap_roc:
        shap_data  = shap_roc[key]["shap"]
        feat_names = shap_data["features"]
        mean_abs   = shap_data["mean_abs"]
        imp_df = pd.DataFrame({"Feature":feat_names,"Mean |SHAP|":mean_abs})
        imp_df = imp_df.sort_values("Mean |SHAP|",ascending=True).tail(12)
        bar_shap = go.Figure(go.Bar(
            x=imp_df["Mean |SHAP|"],y=imp_df["Feature"],orientation="h",
            marker=dict(color=imp_df["Mean |SHAP|"],colorscale=[[0,"#dcfce7"],[0.5,"#2ea85a"],[1,"#064d25"]],showscale=False),
            text=[f"{v:.4f}" for v in imp_df["Mean |SHAP|"]],textposition="outside"))
        bar_shap.update_layout(height=420,margin=dict(t=10,b=10,l=10,r=60),
                               plot_bgcolor="white",paper_bgcolor="white",xaxis_title="Mean |SHAP Value|")
        st.plotly_chart(bar_shap, key="shap_global_bar")
    else:
        st.warning("shap_roc_data.json not found. Place it in the same folder as app.py.")
    st.markdown('</div>',unsafe_allow_html=True)

    # Beeswarm
    if shap_roc:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("🐝 SHAP Beeswarm Plot")
        st.caption("Each dot = one patient. Red = high feature value, Blue = low.")
        sv  = np.array(shap_data["shap_values"])
        fv  = np.array(shap_data["feature_values"])
        order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:10]
        bee_fig = go.Figure()
        rng = np.random.default_rng(42)
        for rank,fi in enumerate(reversed(order)):
            fname  = feat_names[fi]
            x_vals = sv[:,fi]
            fv_col = fv[:,fi]
            fv_norm = (fv_col - fv_col.min()) / ((fv_col.max() - fv_col.min()) + 1e-9)
            y_jit  = rank + rng.uniform(-0.3,0.3,size=len(x_vals))
            colors_bee = [f"rgb({int(255*v)},{int(50+80*(1-v))},{int(255*(1-v))})" for v in fv_norm]
            bee_fig.add_trace(go.Scatter(
                x=x_vals,y=y_jit,mode="markers",
                marker=dict(size=4,color=colors_bee,opacity=0.6),
                name=fname,showlegend=False,
                hovertemplate=f"<b>{fname}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{customdata:.3f}}",
                customdata=fv_col))
        feat_labels = [feat_names[fi] for fi in reversed(order)]
        bee_fig.update_layout(
            height=500,xaxis_title="SHAP Value (impact on model output)",
            yaxis=dict(tickvals=list(range(len(feat_labels))),ticktext=feat_labels),
            plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10,l=10,r=10),
            shapes=[dict(type="line",x0=0,x1=0,y0=-0.5,y1=len(feat_labels)-0.5,
                         line=dict(color="#aaa",dash="dash",width=1))])
        st.plotly_chart(bee_fig, key="shap_beeswarm")
        st.markdown('</div>',unsafe_allow_html=True)

    # Per-patient waterfall
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("🔬 Per-Patient SHAP Explanation")
    if "latest_result" in st.session_state and st.session_state["latest_result"]["Disease"]==disease:
        result      = st.session_state["latest_result"]
        prob        = result["Risk Percentage"]
        level_label = result["Risk Level"]
        st.success(f"Using last prediction — {level_label} ({prob:.2f}%)")
        if shap_roc:
            try:
                import shap as _shap
                if disease=="CKD":
                    enc = {feat:(float(CKD_BINARY_FEATURES[feat].get(str(result.get(feat,0)),result.get(feat,0)))
                                 if feat in CKD_BINARY_FEATURES else float(result.get(feat,0)))
                           for feat in CKD_FEATURES}
                    patient_df = pd.DataFrame([enc],columns=CKD_FEATURES)
                    feat_list  = CKD_FEATURES
                    m = load_model(CKD_MODEL_PATHS["Random Forest"])
                else:
                    patient_df = preprocess_diabetes_input({f:result.get(f,0) for f in DIABETES_FEATURES})
                    feat_list  = DIABETES_FEATURES
                    m = load_model(DIABETES_MODEL_PATHS["XGBoost"])
                clf       = list(m.named_steps.values())[-1]
                explainer = _shap.TreeExplainer(clf)
                sv_raw = explainer.shap_values(patient_df)
                sv_arr = np.array(sv_raw)
                ev     = np.array(explainer.expected_value)
                # CKD RF  -> (1, n_features, 2): take positive class index 1
                # Diabetes XGB -> (1, n_features): single output
                if sv_arr.ndim == 3:
                    sv_patient = sv_arr[0, :, 1]
                    base_val   = float(ev.flat[1])
                else:
                    sv_patient = sv_arr[0]
                    base_val   = float(ev.flat[0])
                contrib_df = pd.DataFrame({"Feature":feat_list,"Value":patient_df.values[0],"SHAP":sv_patient})
                contrib_df = contrib_df.reindex(contrib_df["SHAP"].abs().sort_values(ascending=False).index).head(10).sort_values("SHAP")
                colors_wf  = ["#b91c1c" if v>0 else "#166534" for v in contrib_df["SHAP"]]
                wf_fig = go.Figure(go.Bar(
                    x=contrib_df["SHAP"],
                    y=[f"{r['Feature']} = {r['Value']:.2f}" for _,r in contrib_df.iterrows()],
                    orientation="h",marker_color=colors_wf,
                    text=[f"{v:+.4f}" for v in contrib_df["SHAP"]],textposition="outside"))
                wf_fig.update_layout(
                    height=420,title=f"Base value: {base_val:.3f}  \u2192  Prediction: {prob/100:.3f}",
                    xaxis_title="SHAP contribution (red=toward disease, green=away)",
                    plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=40,b=10,l=10,r=60),
                    shapes=[dict(type="line",x0=0,x1=0,y0=-0.5,y1=len(contrib_df)-0.5,
                                 line=dict(color="#aaa",dash="dash",width=1))])
                st.plotly_chart(wf_fig, key="shap_waterfall")
            except Exception as e:
                st.error(f"Could not compute SHAP: {e}")
    else:
        st.info("💡 Run a prediction first on the Prediction page — then come back here for a patient-specific SHAP explanation.")
    st.markdown('</div>',unsafe_allow_html=True)
    st.info(DISCLAIMER)


elif page == "Model Performance":
    page_header("Model Performance","Real test-set results from trained pipelines.")

    CKD_METRICS = {
        "Logistic Regression":dict(Accuracy=0.8875,Precision=1.0000,Recall=0.8200,F1=0.9011,ROC_AUC=1.0000,Kappa=0.7736,Brier=0.0754),
        "Random Forest":      dict(Accuracy=1.0000,Precision=1.0000,Recall=1.0000,F1=1.0000,ROC_AUC=1.0000,Kappa=1.0000,Brier=0.0103),
        "XGBoost":            dict(Accuracy=1.0000,Precision=1.0000,Recall=1.0000,F1=1.0000,ROC_AUC=1.0000,Kappa=1.0000,Brier=0.0030),
    }
    DIABETES_METRICS = {
        "Logistic Regression":dict(Accuracy=0.7078,Precision=0.5738,Recall=0.6481,F1=0.6087,ROC_AUC=0.8083,Kappa=0.3769,Brier=0.1832),
        "Random Forest":      dict(Accuracy=0.7338,Precision=0.5867,Recall=0.8148,F1=0.6822,ROC_AUC=0.8183,Kappa=0.4634,Brier=0.1713),
        "XGBoost":            dict(Accuracy=0.7338,Precision=0.5915,Recall=0.7778,F1=0.6720,ROC_AUC=0.8172,Kappa=0.4548,Brier=0.1732),
    }
    CKD_CM = {
        "Logistic Regression":[[30,0],[9,41]],
        "Random Forest":      [[30,0],[0,50]],
        "XGBoost":            [[30,0],[0,50]],
    }
    DIABETES_CM = {
        "Logistic Regression":[[74,26],[19,35]],
        "Random Forest":      [[69,31],[10,44]],
        "XGBoost":            [[71,29],[12,42]],
    }

    disease_tab  = st.selectbox("Select Disease",["CKD","Diabetes"],key="perf_disease")
    metrics_dict = CKD_METRICS   if disease_tab=="CKD" else DIABETES_METRICS
    cm_dict      = CKD_CM        if disease_tab=="CKD" else DIABETES_CM
    labels       = ["Not CKD","CKD"] if disease_tab=="CKD" else ["No Diabetes","Diabetes"]

    best_model = max(metrics_dict,key=lambda m:metrics_dict[m]["ROC_AUC"])
    bm = metrics_dict[best_model]
    st.markdown(f"<div class='card'><b>🏆 Best Model (by ROC-AUC): {best_model}</b></div>",unsafe_allow_html=True)
    k_cols = st.columns(5)
    for col,label,km in zip(k_cols,["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],["Accuracy","Precision","Recall","F1","ROC_AUC"]):
        val = bm[km]
        css = "low" if val>=0.85 else ("moderate" if val>=0.70 else "high")
        col.markdown(f'<div class="card" style="text-align:center"><div style="font-size:13px;color:#667085">{label}</div><div class="kpi-value" style="font-size:28px">{val:.4f}</div><span class="{css}">{"Excellent" if val>=0.85 else ("Good" if val>=0.70 else "Fair")}</span></div>',unsafe_allow_html=True)

    # Full table
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("📊 Full Metrics Comparison")
    rows = [{"Model":mn,"Accuracy":f"{m['Accuracy']:.4f}","Precision":f"{m['Precision']:.4f}",
             "Recall":f"{m['Recall']:.4f}","F1-Score":f"{m['F1']:.4f}",
             "ROC-AUC":f"{m['ROC_AUC']:.4f}","Kappa":f"{m['Kappa']:.4f}","Brier \u2193":f"{m['Brier']:.4f}"}
            for mn,m in metrics_dict.items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)

    # Bar chart
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("📈 Metric Bar Chart")
    metric_choice = st.selectbox("Choose metric",["Accuracy","Precision","Recall","F1","ROC_AUC","Kappa"],key="metric_bar")
    bar_fig = go.Figure()
    for i,(mn,m) in enumerate(metrics_dict.items()):
        bar_fig.add_trace(go.Bar(name=mn,x=[mn],y=[m[metric_choice]],marker_color=PALETTE[i],
                                 text=[f"{m[metric_choice]:.4f}"],textposition="outside"))
    bar_fig.update_layout(yaxis=dict(range=[0,1.1],title=metric_choice),
                          plot_bgcolor="white",paper_bgcolor="white",showlegend=False,height=350,margin=dict(t=20,b=20))
    st.plotly_chart(bar_fig, key="perf_bar_chart")
    st.markdown('</div>',unsafe_allow_html=True)

    # ROC curves
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("📉 ROC Curves")
    shap_roc = load_shap_roc()
    if shap_roc:
        roc_key = "ckd" if disease_tab=="CKD" else "diabetes"
        roc_fig = go.Figure()
        for (mn_roc,rdata),color in zip(shap_roc[roc_key]["roc"].items(),PALETTE):
            roc_fig.add_trace(go.Scatter(x=rdata["fpr"],y=rdata["tpr"],mode="lines",
                                         name=f"{mn_roc} (AUC={rdata['auc']:.4f})",line=dict(color=color,width=2.5)))
        roc_fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                     line=dict(color="#aaa",dash="dash",width=1),name="Random Classifier"))
        roc_fig.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",
                              height=420,plot_bgcolor="white",paper_bgcolor="white",
                              legend=dict(x=0.55,y=0.1),margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(roc_fig, key="perf_roc_curves")
    else:
        st.warning("shap_roc_data.json not found next to app.py.")
    st.markdown('</div>',unsafe_allow_html=True)

    # Confusion matrices
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("🔢 Confusion Matrices")
    cm_cols = st.columns(3)
    for idx,(col,(mn_cm,cm)) in enumerate(zip(cm_cols,cm_dict.items())):
        with col:
            st.markdown(f"**{mn_cm}**")
            tn,fp = cm[0]; fn,tp = cm[1]
            heatmap = go.Figure(go.Heatmap(
                z=[[tn,fp],[fn,tp]],
                x=[f"Pred {labels[0]}",f"Pred {labels[1]}"],
                y=[f"True {labels[0]}",f"True {labels[1]}"],
                colorscale=[[0,"#f0fdf4"],[1,"#087331"]],
                text=[[str(tn),str(fp)],[str(fn),str(tp)]],
                texttemplate="%{text}",textfont={"size":18,"color":"black"},showscale=False))
            heatmap.update_layout(height=250,margin=dict(t=10,b=10,l=10,r=10),
                                  paper_bgcolor="white",plot_bgcolor="white")
            st.plotly_chart(heatmap, key=f"cm_{disease_tab}_{idx}")
    st.markdown('</div>',unsafe_allow_html=True)

    # Gauges
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("🎯 ROC-AUC Gauges")
    g_cols = st.columns(3)
    for idx,(col,(mn_g,m)) in enumerate(zip(g_cols,metrics_dict.items())):
        with col:
            g = go.Figure(go.Indicator(
                mode="gauge+number",value=round(m["ROC_AUC"]*100,2),
                title={"text":mn_g,"font":{"size":13}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#087331"},
                       "steps":[{"range":[0,60],"color":"#fee2e2"},{"range":[60,80],"color":"#fef3c7"},{"range":[80,100],"color":"#dcfce7"}]}))
            g.update_layout(height=220,margin=dict(t=30,b=10,l=10,r=10))
            st.plotly_chart(g, key=f"gauge_{disease_tab}_{idx}")
    st.markdown('</div>',unsafe_allow_html=True)

    # Overfitting
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("🔍 CV vs Hold-out AUC (Overfitting Check)")
    if disease_tab=="CKD":
        ov = {"Model":["Logistic Regression","Random Forest","XGBoost"],
              "CV AUC":[1.0000,0.9998,0.9988],"Test AUC":[1.0000,1.0000,1.0000],
              "Gap":["+0.0000","-0.0002","-0.0012"],"Status":["✅ OK","✅ OK","✅ OK"]}
    else:
        ov = {"Model":["Logistic Regression","Random Forest","XGBoost"],
              "CV AUC":[0.8443,0.8397,0.8359],"Test AUC":[0.8083,0.8183,0.8172],
              "Gap":["+0.0359","+0.0214","+0.0187"],"Status":["✅ OK","✅ OK","✅ OK"]}
    st.dataframe(pd.DataFrame(ov).set_index("Model"),use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)
    st.info(DISCLAIMER)


elif page == "Export / Report":
    page_header("Export / Report","Download latest prediction result.")
    if "latest_result" in st.session_state:
        result_df = pd.DataFrame([st.session_state["latest_result"]])
        st.dataframe(result_df,use_container_width=True)
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV",csv,"predictcare_result.csv","text/csv")
    else:
        st.warning("No prediction result found yet. Go to Prediction page first.")
    st.error(DISCLAIMER)