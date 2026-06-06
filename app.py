import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# PAGE CONFIGURATION
# ======================================
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
)

# ======================================
# LOAD FILES
# FIX: use st.cache_resource so the pipeline and data are loaded
# only once per session, not on every interaction.
# ======================================
@st.cache_resource
def load_pipeline():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("heart_cleaned.csv")

pipeline = load_pipeline()
df = load_data()

# ======================================
# LABEL MAPS
# FIX: raw integer selectors for cp, restecg, slope, thal, ca gave
# users no context. Replaced with descriptive labels mapped to ints.
# ======================================
CP_MAP = {
    "Typical Angina (0)": 0,
    "Atypical Angina (1)": 1,
    "Non-Anginal Pain (2)": 2,
    "Asymptomatic (3)": 3,
}

RESTECG_MAP = {
    "Normal (0)": 0,
    "ST-T Wave Abnormality (1)": 1,
    "Left Ventricular Hypertrophy (2)": 2,
}

SLOPE_MAP = {
    "Upsloping (0)": 0,
    "Flat (1)": 1,
    "Downsloping (2)": 2,
}

THAL_MAP = {
    "Normal (0)": 0,
    "Fixed Defect (1)": 1,
    "Reversible Defect (2)": 2,
    "Unknown (3)": 3,
}

# ======================================
# SIDEBAR NAVIGATION
# ======================================
st.sidebar.title("❤️ Heart Disease System")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "❤️ Prediction", "📊 Dataset", "ℹ️ About"],
)

# ======================================
# HOME PAGE
# ======================================
if page == "🏠 Home":

    st.title("❤️ Heart Disease Prediction Dashboard")
    st.markdown(
        "Welcome to the AI-powered Heart Disease Prediction System. "
        "This application uses Machine Learning to assess heart disease risk."
    )
    st.success("System Ready")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patients", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Target Classes", df["target"].nunique())
    with col4:
        st.metric("Model", "Random Forest")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("Target Distribution")

    # FIX: avoid deprecated pandas .plot() on axes; use explicit matplotlib
    target_counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        ["0 – No Disease", "1 – Disease"],
        target_counts.values,
        color=["steelblue", "tomato"],
    )
    ax.set_ylabel("Count")
    ax.set_title("Target Distribution")
    st.pyplot(fig)
    plt.close(fig)

# ======================================
# PREDICTION PAGE
# ======================================
elif page == "❤️ Prediction":

    st.title("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=50,
        )

        sex_option = st.selectbox("Sex", ["Female (0)", "Male (1)"])
        sex = 1 if sex_option.startswith("Male") else 0

        cp_option = st.selectbox("Chest Pain Type", list(CP_MAP.keys()))
        cp = CP_MAP[cp_option]

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=int(df["trestbps"].min()),
            max_value=int(df["trestbps"].max()),
            value=120,
        )

        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=int(df["chol"].min()),
            max_value=int(df["chol"].max()),
            value=200,
        )

        fbs_option = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
        fbs = 1 if fbs_option.startswith("Yes") else 0

        restecg_option = st.selectbox("Resting ECG Results", list(RESTECG_MAP.keys()))
        restecg = RESTECG_MAP[restecg_option]

    with col2:
        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=int(df["thalach"].min()),
            max_value=int(df["thalach"].max()),
            value=150,
        )

        exang_option = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        exang = 1 if exang_option.startswith("Yes") else 0

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=float(df["oldpeak"].min()),
            max_value=float(df["oldpeak"].max()),
            value=1.0,
            step=0.1,
        )

        slope_option = st.selectbox("Slope of Peak Exercise ST Segment", list(SLOPE_MAP.keys()))
        slope = SLOPE_MAP[slope_option]

        # FIX: derive max ca from actual dataset instead of hardcoding
        ca = st.selectbox(
            "Number of Major Vessels (0–3)",
            list(range(int(df["ca"].max()) + 1)),
        )

        thal_option = st.selectbox("Thalassemia", list(THAL_MAP.keys()))
        thal = THAL_MAP[thal_option]

    # ======================================
    # PREDICTION BUTTON
    # ======================================
    if st.button("🔍 Predict"):

        patient_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal],
        })

        # FIX: pipeline handles scaling internally — no separate scaler needed
        prediction = pipeline.predict(patient_data)[0]

        # FIX: catch AttributeError specifically instead of bare except
        try:
            probability = pipeline.predict_proba(patient_data)[0][1]
            risk = round(probability * 100, 2)
            no_risk = round((1 - probability) * 100, 2)
        except AttributeError:
            risk = None
            no_risk = None

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            msg = f"⚠️ Heart Disease Detected" + (f" — Risk: {risk}%" if risk is not None else "")
            st.error(msg)
        else:
            msg = f"✅ No Heart Disease Detected" + (f" — Confidence: {no_risk}%" if no_risk is not None else "")
            st.success(msg)

        # Risk probability bar
        if risk is not None:
            st.subheader("Risk Probability")
            col_a, col_b = st.columns(2)
            col_a.metric("Disease Probability", f"{risk}%")
            col_b.metric("No Disease Probability", f"{no_risk}%")
            st.progress(int(risk))

        st.subheader("Patient Summary")
        summary = pd.DataFrame({
            "Feature": patient_data.columns,
            "Value": patient_data.iloc[0].values,
        })
        st.dataframe(summary, use_container_width=True)

        st.caption(
            "⚠️ This tool is for educational purposes only and is NOT a medical diagnostic tool. "
            "Always consult a qualified healthcare professional."
        )

# ======================================
# DATASET PAGE
# ======================================
elif page == "📊 Dataset":

    st.title("📊 Dataset Explorer")

    st.dataframe(df)

    st.markdown("---")
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.markdown("---")
    st.subheader("Feature Distribution")

    feature = st.selectbox("Choose Feature", df.columns[:-1])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[feature], bins=20, color="steelblue", edgecolor="white")
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Feature vs Target")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for target_val, label in [(0, "No Disease"), (1, "Disease")]:
        subset = df[df["target"] == target_val][feature]
        ax2.hist(subset, bins=15, alpha=0.6, label=label)
    ax2.set_title(f"{feature} by Target")
    ax2.set_xlabel(feature)
    ax2.set_ylabel("Count")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

# ======================================
# ABOUT PAGE
# ======================================
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    This Heart Disease Prediction System uses Machine Learning
    to predict the likelihood of heart disease based on medical inputs.

    ### Model
    - **Algorithm**: Random Forest Classifier
    - **Pipeline**: StandardScaler → RandomForestClassifier
    - **Validation**: 5-fold Stratified Cross-Validation

    ### Features Used
    | Feature | Description |
    |---|---|
    | age | Age in years |
    | sex | Sex (0 = Female, 1 = Male) |
    | cp | Chest pain type (0–3) |
    | trestbps | Resting blood pressure (mm Hg) |
    | chol | Serum cholesterol (mg/dl) |
    | fbs | Fasting blood sugar > 120 mg/dl |
    | restecg | Resting ECG results |
    | thalach | Maximum heart rate achieved |
    | exang | Exercise-induced angina |
    | oldpeak | ST depression induced by exercise |
    | slope | Slope of peak exercise ST segment |
    | ca | Number of major vessels (0–3) |
    | thal | Thalassemia type |

    ### Disclaimer
    > **This is NOT a medical diagnostic tool.**
    > It is for educational and demonstration purposes only.
    > Always consult a qualified healthcare professional for medical advice.
    """)

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.caption("Developed with Streamlit ❤️")