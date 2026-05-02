import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="PredictCare", page_icon="💚", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CKD_MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODEL_DIR, "CKD", "ckd_lr_pipeline.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "CKD", "ckd_rf_pipeline.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "CKD", "ckd_xgb_pipeline.pkl"),
}

DIABETES_MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODEL_DIR, "diabetes", "diabetes_lr_pipeline.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "diabetes", "diabetes_rf_pipeline.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "diabetes", "diabetes_xgb_pipeline.pkl"),
}

DISCLAIMER = """
⚠️ This system is only for educational and research purposes.
It is not a medical diagnosis tool. Please consult a doctor for medical decisions.
"""

st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #f7fbf8, #ffffff);
    color: #0f172a;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #064d25, #033d1d);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    margin-bottom: 18px;
}

.logo {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 25px;
}

.logo span {
    color: #9df071;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
}

.hero-subtitle {
    color: #667085;
    font-size: 16px;
    margin-bottom: 25px;
}

.kpi-value {
    font-size: 40px;
    font-weight: 800;
    color: #087331;
}

.low {
    color: #166534;
    background: #dcfce7;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}

.moderate {
    color: #b45309;
    background: #fef3c7;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}

.high {
    color: #b91c1c;
    background: #fee2e2;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}

.stButton button {
    background: #087331 !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD DATA
# =========================
@st.cache_data
def find_csv(folder):
    files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    return files[0] if files else None


@st.cache_data
def load_dataset(disease):
    if disease == "CKD":
        path = find_csv(os.path.join(DATA_DIR, "Chronic_Kidney_Disease"))
    else:
        path = find_csv(os.path.join(DATA_DIR, "Diabetes"))

    if path is None:
        st.error(f"No CSV file found for {disease}")
        st.stop()

    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()

    return joblib.load(path)


def get_target_column(df):
    possible_targets = ["classification", "Outcome", "target", "label", "class"]

    for col in possible_targets:
        if col in df.columns:
            return col

    return df.columns[-1]


def get_features(df):
    target = get_target_column(df)
    return [col for col in df.columns if col != target]


# =========================
# IMPORTANT FIX FUNCTION
# =========================
def predict_probability(model, row):
    """
    Correctly gets probability of positive disease class.
    Prevents wrong 0/100 output caused by predict().
    """

    if not hasattr(model, "predict_proba"):
        prediction = model.predict(row)[0]
        return float(prediction) * 100

    proba = model.predict_proba(row)[0]

    if hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]
    else:
        final_model = model

    if hasattr(final_model, "classes_"):
        classes = list(final_model.classes_)
    else:
        classes = [0, 1]

    positive_labels = [1, "1", "yes", "Yes", "YES", "ckd", "CKD", "diabetes", "Diabetes"]

    positive_index = None

    for label in positive_labels:
        if label in classes:
            positive_index = classes.index(label)
            break

    if positive_index is None:
        positive_index = len(proba) - 1

    probability = float(proba[positive_index]) * 100

    return probability


def risk_level(prob):
    if prob >= 70:
        return "High Risk", "high"
    elif prob >= 40:
        return "Moderate Risk", "moderate"
    else:
        return "Low Risk", "low"


def get_feature_importance(model, features):
    if hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]
    else:
        final_model = model

    if hasattr(final_model, "feature_importances_"):
        values = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        values = np.abs(final_model.coef_).ravel()
    else:
        values = np.ones(len(features))

    values = values[:len(features)]

    return pd.DataFrame({
        "Feature": features[:len(values)],
        "Importance": values
    }).sort_values("Importance", ascending=False)


def explain_risk_causes(disease, input_values, importance_df):
    diabetes_rules = {
        "Glucose": "High glucose is one of the strongest diabetes risk factors.",
        "BMI": "High BMI can increase diabetes risk.",
        "Age": "Older age can increase diabetes risk.",
        "Insulin": "Abnormal insulin can affect blood sugar control.",
        "BloodPressure": "High blood pressure can increase diabetes-related risk.",
        "DiabetesPedigreeFunction": "Family history can increase diabetes risk.",
        "Pregnancies": "Pregnancy count may influence diabetes risk.",
        "SkinThickness": "Skin thickness may relate to body fat level."
    }

    ckd_rules = {
        "sc": "High serum creatinine may indicate reduced kidney function.",
        "bu": "High blood urea may suggest kidney filtering problems.",
        "al": "Albumin in urine may indicate kidney damage.",
        "hemo": "Low hemoglobin may be linked with CKD-related anemia.",
        "bp": "High blood pressure can damage kidney function.",
        "bgr": "High blood glucose can increase kidney disease risk.",
        "htn": "Hypertension history increases CKD risk.",
        "dm": "Diabetes history increases CKD risk.",
        "sg": "Abnormal specific gravity may suggest kidney concentration problems.",
        "pcv": "Low packed cell volume may be linked with kidney disease.",
        "ane": "Anemia may be associated with CKD.",
        "pe": "Pedal edema may be related to kidney problems."
    }

    rules = diabetes_rules if disease == "Diabetes" else ckd_rules

    output = []

    for _, row in importance_df.head(5).iterrows():
        feature = row["Feature"]
        value = input_values.get(feature, "N/A")

        output.append({
            "Feature": feature,
            "Patient Value": value,
            "Importance": row["Importance"],
            "Reason": rules.get(feature, "This feature strongly influenced the model prediction.")
        })

    return output


def page_header(title, subtitle):
    st.markdown(f"""
    <div>
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================
st.sidebar.markdown(
    '<div class="logo">💚 Predict<span>Care</span></div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard / Overview",
        "Prediction",
        "Explainability",
        "Model Performance",
        "Export / Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(DISCLAIMER)


# =========================
# DASHBOARD
# =========================
if page == "Dashboard / Overview":
    page_header(
        "Dashboard Overview",
        "AI-powered disease risk prediction for CKD and Diabetes."
    )

    ckd_df = load_dataset("CKD")
    diabetes_df = load_dataset("Diabetes")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>CKD Records</h4>
            <div class="kpi-value">{len(ckd_df)}</div>
            <span class="low">Dataset Ready</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>Diabetes Records</h4>
            <div class="kpi-value">{len(diabetes_df)}</div>
            <span class="low">Dataset Ready</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h4>Model Status</h4>
            <div class="kpi-value">Ready</div>
            <span class="low">6 Models</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_features = len(get_features(ckd_df)) + len(get_features(diabetes_df))
        st.markdown(f"""
        <div class="card">
            <h4>Total Features</h4>
            <div class="kpi-value">{total_features}</div>
            <span class="moderate">Clinical Inputs</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset Summary")
    st.write("This dashboard supports CKD and Diabetes prediction using machine learning pipelines.")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# PREDICTION
# =========================
elif page == "Prediction":
    page_header(
        "Prediction Page",
        "Enter patient values and predict disease risk."
    )

    disease = st.selectbox("Select Disease", ["CKD", "Diabetes"])
    model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    df = load_dataset(disease)
    features = get_features(df)

    if disease == "CKD":
        model_path = CKD_MODEL_PATHS[model_name]
    else:
        model_path = DIABETES_MODEL_PATHS[model_name]

    model = load_model(model_path)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient Input Form")

    input_values = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        with cols[i % 3]:
            if pd.api.types.is_numeric_dtype(df[feature]):
                input_values[feature] = st.number_input(
                    feature,
                    value=float(df[feature].median()),
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max())
                )
            else:
                input_values[feature] = st.selectbox(
                    feature,
                    options=df[feature].dropna().unique()
                )

    predict_btn = st.button("Predict Risk")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        row = pd.DataFrame([input_values], columns=features)

        probability = predict_probability(model, row)
        level, css_class = risk_level(probability)

        st.session_state["latest_result"] = {
            "Disease": disease,
            "Model": model_name,
            "Risk Percentage": round(probability, 2),
            "Risk Level": level,
            **input_values
        }

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="card">
                <h2>Prediction Result</h2>
                <div class="kpi-value">{probability:.2f}%</div>
                <span class="{css_class}">{level}</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                title={"text": "Risk Percentage"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#087331"},
                    "steps": [
                        {"range": [0, 40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fee2e2"}
                    ]
                }
            ))

            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DEBUG OUTPUT
        # =========================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Debug Check")

        if hasattr(model, "named_steps"):
            final_model = list(model.named_steps.values())[-1]
        else:
            final_model = model

        if hasattr(final_model, "classes_"):
            st.write("Model classes:", list(final_model.classes_))

        if hasattr(model, "predict_proba"):
            st.write("Raw predict_proba output:")
            st.write(model.predict_proba(row))

        st.write("Raw predict output:")
        st.write(model.predict(row))

        st.info(
            "If predict_proba shows [[0, 1]] or similar for every input, "
            "your trained model is predicting disease for everything and needs retraining."
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # WHAT CAUSES RISK
        # =========================
        importance_df = get_feature_importance(model, features).head(5)

        risk_causes = explain_risk_causes(
            disease=disease,
            input_values=input_values,
            importance_df=importance_df
        )

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


# =========================
# EXPLAINABILITY
# =========================
elif page == "Explainability":
    page_header(
        "Explainability Page",
        "View feature importance and risk explanation."
    )

    disease = st.selectbox("Select Disease", ["CKD", "Diabetes"])
    model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    df = load_dataset(disease)
    features = get_features(df)

    model_path = CKD_MODEL_PATHS[model_name] if disease == "CKD" else DIABETES_MODEL_PATHS[model_name]
    model = load_model(model_path)

    importance_df = get_feature_importance(model, features).head(10)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature Importance")
    st.dataframe(importance_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Risk Explanation")

    if disease == "Diabetes":
        st.write("""
        Diabetes risk may increase due to high glucose, high BMI, older age,
        abnormal insulin level, high blood pressure, and family history.
        """)
    else:
        st.write("""
        CKD risk may increase due to high serum creatinine, high blood urea,
        albumin in urine, low hemoglobin, high blood pressure, diabetes history,
        and hypertension history.
        """)

    st.info(DISCLAIMER)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# MODEL PERFORMANCE
# =========================
elif page == "Model Performance":
    page_header(
        "Model Performance Page",
        "Model quality summary."
    )

    st.info("You can add Accuracy, F1-score, ROC-AUC and confusion matrix here.")


# =========================
# EXPORT
# =========================
elif page == "Export / Report":
    page_header(
        "Export / Report Page",
        "Download latest prediction result."
    )

    if "latest_result" in st.session_state:
        result_df = pd.DataFrame([st.session_state["latest_result"]])
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            csv,
            "predictcare_prediction_result.csv",
            "text/csv"
        )
    else:
        st.warning("No prediction result found yet. Go to Prediction page first.")

    st.error(DISCLAIMER)