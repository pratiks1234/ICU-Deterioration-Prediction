from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


DATA_PATH = Path(
    "data/processed/icu_cohort_with_vitals_labs_24h.csv"
)

OOF_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_predictions_tuned.csv"
)

THRESHOLD_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_threshold_summary.json"
)

MODEL_PATH = Path(
    "outputs/models/xgboost_reduced_clinical_final.joblib"
)


FEATURE_COLUMNS = [
    "gender",
    "anchor_age",
    "admission_type",
    "first_careunit",
    "heart_rate_mean_24h",
    "heart_rate_max_24h",
    "resp_rate_mean_24h",
    "spo2_min_24h",
    "map_min_24h",
    "temperature_max_24h",
    "lactate_max_24h",
    "creatinine_latest_24h",
    "bun_latest_24h",
    "wbc_max_24h",
    "platelets_min_24h",
    "bicarbonate_min_24h",
]


VITAL_COLUMNS = [
    "heart_rate_mean_24h",
    "heart_rate_max_24h",
    "resp_rate_mean_24h",
    "spo2_min_24h",
    "map_min_24h",
    "temperature_max_24h",
]


LAB_COLUMNS = [
    "lactate_max_24h",
    "creatinine_latest_24h",
    "bun_latest_24h",
    "wbc_max_24h",
    "platelets_min_24h",
    "bicarbonate_min_24h",
]


st.set_page_config(
    page_title="ICU Clinical Risk Dashboard",
    page_icon="🏥",
    layout="wide",
)


@st.cache_data
def load_dashboard_data():
    cohort = pd.read_csv(DATA_PATH)
    oof = pd.read_csv(OOF_PATH)

    oof_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "fold",
        "predicted_risk",
        "predicted_class_tuned",
    ]

    return cohort.merge(
        oof[oof_columns],
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_thresholds():
    with open(THRESHOLD_PATH) as file:
        summary = json.load(file)

    high_threshold = float(summary["recommended_threshold"])
    medium_threshold = float(
        summary["high_recall_result"]["threshold"]
    )

    return medium_threshold, high_threshold


def risk_category(score, medium_threshold, high_threshold):
    if score < medium_threshold:
        return "Low Risk"

    if score < high_threshold:
        return "Medium Risk"

    return "High Risk"


def prepare_shap_explanation(pipeline, patient_features):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(patient_features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": transformed[0],
            "shap_value": shap_values[0],
        }
    )

    explanation["absolute_shap"] = explanation["shap_value"].abs()

    return explanation.sort_values(
        "absolute_shap",
        ascending=False,
    )


def clean_feature_name(name):
    name = name.replace("numeric__", "")
    name = name.replace("categorical__", "")
    name = name.replace("_24h", "")
    name = name.replace("_", " ")

    return name.title()


def main():
    st.title("ICU Patient Clinical Risk Dashboard")

    required_files = [
        DATA_PATH,
        OOF_PATH,
        THRESHOLD_PATH,
        MODEL_PATH,
    ]

    for path in required_files:
        if not path.exists():
            st.error(f"Missing required file: {path}")
            st.stop()

    dashboard_df = load_dashboard_data()
    pipeline = load_model()

    medium_threshold, high_threshold = load_thresholds()

    dashboard_df = dashboard_df.sort_values(
        "predicted_risk",
        ascending=False,
    ).reset_index(drop=True)

    dashboard_df["patient_option"] = dashboard_df.apply(
        lambda row: (
            f"subject_id={int(row['subject_id'])} | "
            f"hadm_id={int(row['hadm_id'])} | "
            f"stay_id={int(row['stay_id'])} | "
            f"risk={row['predicted_risk'] * 100:.1f}%"
        ),
        axis=1,
    )

    st.sidebar.header("Patient Selection")

    selected_option = st.sidebar.selectbox(
        "Select ICU stay",
        dashboard_df["patient_option"],
    )

    selected_patient = dashboard_df[
        dashboard_df["patient_option"] == selected_option
    ].iloc[0]

    patient_features = selected_patient[
        FEATURE_COLUMNS
    ].to_frame().T

    final_model_risk = pipeline.predict_proba(
        patient_features
    )[0, 1]

    oof_risk = float(selected_patient["predicted_risk"])

    category = risk_category(
        oof_risk,
        medium_threshold,
        high_threshold,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Out-of-Fold Risk",
        f"{oof_risk * 100:.2f}%",
        help=(
            "This prediction was produced by a model that did not "
            "train on this ICU stay."
        ),
    )

    col2.metric("Risk Category", category)

    col3.metric(
        "Actual Outcome",
        int(selected_patient["hospital_expire_flag"]),
        help="0 = survived, 1 = hospital mortality",
    )

    col4.metric(
        "Final Model Risk",
        f"{final_model_risk * 100:.2f}%",
        help=(
            "This score comes from the final model trained on the "
            "complete demo dataset."
        ),
    )

    st.caption(
        f"Medium-risk threshold: {medium_threshold:.2f} | "
        f"High-risk threshold: {high_threshold:.2f} | "
        f"Validation fold: {int(selected_patient['fold'])}"
    )

    st.subheader("Patient Information")

    patient_info = {
        "Subject ID": int(selected_patient["subject_id"]),
        "Hospital Admission ID": int(selected_patient["hadm_id"]),
        "ICU Stay ID": int(selected_patient["stay_id"]),
        "Gender": selected_patient["gender"],
        "Age": selected_patient["anchor_age"],
        "Admission Type": selected_patient["admission_type"],
        "First ICU Unit": selected_patient["first_careunit"],
    }

    st.dataframe(
        pd.DataFrame.from_dict(
            patient_info,
            orient="index",
            columns=["Value"],
        ),
        use_container_width=True,
    )

    vital_col, lab_col = st.columns(2)

    with vital_col:
        st.subheader("First 24-Hour Vitals")

        vital_table = selected_patient[
            VITAL_COLUMNS
        ].rename(index=clean_feature_name).to_frame("Value")

        st.dataframe(
            vital_table,
            use_container_width=True,
        )

    with lab_col:
        st.subheader("First 24-Hour Labs")

        lab_table = selected_patient[
            LAB_COLUMNS
        ].rename(index=clean_feature_name).to_frame("Value")

        st.dataframe(
            lab_table,
            use_container_width=True,
        )

    st.subheader("Clinical Warning")

    if category == "High Risk":
        st.error(
            "This ICU stay is classified as high risk using the "
            "tuned clinical threshold."
        )

    elif category == "Medium Risk":
        st.warning(
            "This ICU stay is classified as medium risk. "
            "Closer monitoring may be appropriate."
        )

    else:
        st.success(
            "This ICU stay is classified as low risk by the model."
        )

    st.subheader("SHAP Explanation")

    st.caption(
        "The SHAP explanation below uses the final model. "
        "Positive values increase predicted risk and negative values "
        "decrease predicted risk."
    )

    shap_df = prepare_shap_explanation(
        pipeline,
        patient_features,
    )

    shap_df["feature"] = shap_df["feature"].apply(
        clean_feature_name
    )

    risk_increasing = shap_df[
        shap_df["shap_value"] > 0
    ].head(8)

    risk_decreasing = shap_df[
        shap_df["shap_value"] < 0
    ].head(8)

    increase_col, decrease_col = st.columns(2)

    with increase_col:
        st.markdown("#### Risk-Increasing Features")
        st.dataframe(
            risk_increasing[
                ["feature", "feature_value", "shap_value"]
            ],
            use_container_width=True,
        )

    with decrease_col:
        st.markdown("#### Risk-Decreasing Features")
        st.dataframe(
            risk_decreasing[
                ["feature", "feature_value", "shap_value"]
            ],
            use_container_width=True,
        )

    st.info(
        "This dashboard is a research demonstration using the "
        "MIMIC-IV demo dataset and is not intended for clinical use."
    )


if __name__ == "__main__":
    main()
