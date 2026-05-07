from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st


DATA_PATH = Path("data/processed/icu_cohort_with_vitals_24h.csv")
MODEL_PATH = Path("outputs/models/xgboost_vitals_24h_mortality.joblib")


st.set_page_config(
    page_title="ICU Risk Prediction Dashboard",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    saved_model = joblib.load(MODEL_PATH)
    return saved_model["model"], saved_model["feature_columns"]


def prepare_features(df, saved_features):
    base_features = [
        "gender",
        "anchor_age",
        "admission_type",
        "admission_location",
        "insurance",
        "race",
        "first_careunit",
    ]

    vital_features = [col for col in df.columns if col.endswith("_24h")]
    feature_cols = base_features + vital_features

    X = df[feature_cols].copy()

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    X[categorical_cols] = X[categorical_cols].fillna("Unknown")

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.reindex(columns=saved_features, fill_value=0)

    return X_encoded


def risk_category(score):
    if score < 0.10:
        return "Low Risk"
    elif score < 0.30:
        return "Medium Risk"
    return "High Risk"


def main():
    st.title("ICU Patient Risk Prediction Dashboard")

    if not DATA_PATH.exists():
        st.error(f"Missing data file: {DATA_PATH}")
        st.stop()

    if not MODEL_PATH.exists():
        st.error(f"Missing model file: {MODEL_PATH}")
        st.stop()

    df = load_data()
    model, saved_features = load_model()

    X_encoded = prepare_features(df, saved_features)
    risk_scores = model.predict_proba(X_encoded)[:, 1]

    df_display = df.copy()
    df_display["predicted_risk"] = risk_scores

    st.sidebar.header("Patient Selection")

    patient_options = df_display.apply(
        lambda row: f"subject_id={row['subject_id']} | hadm_id={row['hadm_id']} | stay_id={row['stay_id']}",
        axis=1,
    )

    selected_option = st.sidebar.selectbox(
        "Select ICU stay",
        patient_options,
    )

    selected_index = patient_options[patient_options == selected_option].index[0]
    selected_patient = df_display.loc[selected_index]
    selected_features = X_encoded.loc[[selected_index]]

    risk = selected_patient["predicted_risk"]
    category = risk_category(risk)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Risk", f"{risk * 100:.2f}%")

    with col2:
        st.metric("Risk Category", category)

    with col3:
        if "hospital_expire_flag" in df_display.columns:
            st.metric("Actual Outcome", int(selected_patient["hospital_expire_flag"]))

    st.subheader("Patient Information")

    info_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "gender",
        "anchor_age",
        "admission_type",
        "admission_location",
        "insurance",
        "race",
        "first_careunit",
        "last_careunit",
    ]

    existing_info_cols = [col for col in info_cols if col in df_display.columns]
    st.dataframe(selected_patient[existing_info_cols].to_frame("Value"))

    st.subheader("24-hour Vitals Summary")

    vital_cols = [
        col for col in df_display.columns
        if col.endswith("_24h") and not col.endswith("_missing_24h")
    ]

    vital_summary = selected_patient[vital_cols].dropna().to_frame("Value")
    st.dataframe(vital_summary)

    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(selected_features)

    shap_df = pd.DataFrame(
        {
            "feature": selected_features.columns,
            "feature_value": selected_features.iloc[0].values,
            "shap_value": shap_values[0],
        }
    )

    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False)

    st.write(
        "Positive SHAP values push the model toward higher risk. "
        "Negative SHAP values push the model toward lower risk."
    )

    st.dataframe(shap_df.head(15))

    st.subheader("Clinical Warning Explanation")

    top_positive = shap_df[shap_df["shap_value"] > 0].head(5)

    if category == "High Risk":
        st.warning(
            "This patient is predicted to be high risk. "
            "The main contributing features are shown below."
        )
    elif category == "Medium Risk":
        st.info(
            "This patient is predicted to be medium risk. "
            "Clinical monitoring may be needed."
        )
    else:
        st.success(
            "This patient is predicted to be low risk based on the current model."
        )

    st.dataframe(top_positive[["feature", "feature_value", "shap_value"]])


if __name__ == "__main__":
    main()
