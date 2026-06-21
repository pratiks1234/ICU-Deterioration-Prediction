from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "deployment"
    / "xgboost_reduced_clinical_final.joblib"
)

MEDIUM_THRESHOLD = 0.21
HIGH_THRESHOLD = 0.38


st.set_page_config(
    page_title="ICU Mortality Risk Demo",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Deployment model was not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def get_category_options(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]

    categorical_columns = []
    categorical_transformer = None

    for name, transformer, columns in preprocessor.transformers_:
        if name == "categorical":
            categorical_columns = list(columns)
            categorical_transformer = transformer
            break

    if categorical_transformer is None:
        return {}

    encoder = categorical_transformer.named_steps["onehot"]

    return {
        column: [str(value) for value in categories]
        for column, categories in zip(
            categorical_columns,
            encoder.categories_,
        )
    }


def risk_category(probability):
    if probability < MEDIUM_THRESHOLD:
        return "Low Risk"

    if probability < HIGH_THRESHOLD:
        return "Medium Risk"

    return "High Risk"


def clean_feature_name(name):
    return (
        name.replace("numeric__", "")
        .replace("categorical__", "")
        .replace("_24h", "")
        .replace("_", " ")
        .title()
    )


def create_shap_table(pipeline, input_data):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(input_data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        elif shap_values.shape[0] == 2:
            shap_values = shap_values[1]

    feature_names = preprocessor.get_feature_names_out()

    results = pd.DataFrame(
        {
            "Feature": [
                clean_feature_name(name)
                for name in feature_names
            ],
            "Input Value": transformed[0],
            "SHAP Value": shap_values[0],
        }
    )

    results["Absolute SHAP"] = results["SHAP Value"].abs()
    results["Effect"] = np.where(
        results["SHAP Value"] >= 0,
        "Increases model risk",
        "Decreases model risk",
    )

    return results.sort_values(
        "Absolute SHAP",
        ascending=False,
    )


def main():
    st.title("ICU Hospital Mortality Risk Demonstration")

    st.write(
        "Enter simulated first-24-hour ICU clinical measurements "
        "to generate a mortality-risk estimate from the reduced "
        "XGBoost model."
    )

    st.warning(
        "Research and educational demonstration only. "
        "This application is not a medical device and must not be "
        "used for diagnosis, treatment, monitoring, or clinical "
        "decision-making."
    )

    st.info(
        "Do not enter names, medical-record numbers, dates of birth, "
        "or any other identifiable patient information."
    )

    pipeline = load_model()
    category_options = get_category_options(pipeline)

    gender_options = category_options.get(
        "gender",
        ["F", "M"],
    )

    admission_options = category_options.get(
        "admission_type",
        ["EW EMER.", "URGENT", "ELECTIVE"],
    )

    careunit_options = category_options.get(
        "first_careunit",
        [
            "Medical Intensive Care Unit (MICU)",
            "Surgical Intensive Care Unit (SICU)",
        ],
    )

    with st.form("clinical_input_form"):
        st.subheader("Patient and Admission Information")

        patient_col1, patient_col2 = st.columns(2)

        with patient_col1:
            gender = st.selectbox(
                "Gender",
                gender_options,
            )

            anchor_age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=65,
                step=1,
            )

        with patient_col2:
            admission_type = st.selectbox(
                "Admission type",
                admission_options,
            )

            first_careunit = st.selectbox(
                "First ICU care unit",
                careunit_options,
            )

        st.subheader("First 24-Hour Vital Signs")

        vital_col1, vital_col2, vital_col3 = st.columns(3)

        with vital_col1:
            heart_rate_mean = st.number_input(
                "Mean heart rate (bpm)",
                min_value=20.0,
                max_value=220.0,
                value=85.0,
                step=1.0,
            )

            heart_rate_max = st.number_input(
                "Maximum heart rate (bpm)",
                min_value=20.0,
                max_value=260.0,
                value=110.0,
                step=1.0,
            )

        with vital_col2:
            resp_rate_mean = st.number_input(
                "Mean respiratory rate",
                min_value=4.0,
                max_value=70.0,
                value=20.0,
                step=1.0,
            )

            spo2_min = st.number_input(
                "Minimum oxygen saturation (%)",
                min_value=40.0,
                max_value=100.0,
                value=92.0,
                step=1.0,
            )

        with vital_col3:
            map_min = st.number_input(
                "Minimum mean arterial pressure (mmHg)",
                min_value=20.0,
                max_value=140.0,
                value=65.0,
                step=1.0,
            )

            temperature_max = st.number_input(
                "Maximum temperature (°C)",
                min_value=30.0,
                max_value=43.0,
                value=38.0,
                step=0.1,
            )

        st.subheader("First 24-Hour Laboratory Results")

        lab_col1, lab_col2, lab_col3 = st.columns(3)

        with lab_col1:
            lactate_max = st.number_input(
                "Maximum lactate (mmol/L)",
                min_value=0.0,
                max_value=30.0,
                value=2.0,
                step=0.1,
            )

            creatinine_latest = st.number_input(
                "Latest creatinine (mg/dL)",
                min_value=0.1,
                max_value=20.0,
                value=1.2,
                step=0.1,
            )

        with lab_col2:
            bun_latest = st.number_input(
                "Latest BUN (mg/dL)",
                min_value=1.0,
                max_value=250.0,
                value=25.0,
                step=1.0,
            )

            wbc_max = st.number_input(
                "Maximum WBC (K/µL)",
                min_value=0.1,
                max_value=100.0,
                value=12.0,
                step=0.5,
            )

        with lab_col3:
            platelets_min = st.number_input(
                "Minimum platelets (K/µL)",
                min_value=1.0,
                max_value=1000.0,
                value=180.0,
                step=5.0,
            )

            bicarbonate_min = st.number_input(
                "Minimum bicarbonate (mEq/L)",
                min_value=1.0,
                max_value=60.0,
                value=24.0,
                step=1.0,
            )

        submitted = st.form_submit_button(
            "Generate Risk Estimate",
            use_container_width=True,
        )

    if not submitted:
        st.caption(
            "Default values are illustrative and do not represent "
            "a real patient."
        )
        return

    input_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "anchor_age": float(anchor_age),
                "admission_type": admission_type,
                "first_careunit": first_careunit,
                "heart_rate_mean_24h": float(heart_rate_mean),
                "heart_rate_max_24h": float(heart_rate_max),
                "resp_rate_mean_24h": float(resp_rate_mean),
                "spo2_min_24h": float(spo2_min),
                "map_min_24h": float(map_min),
                "temperature_max_24h": float(temperature_max),
                "lactate_max_24h": float(lactate_max),
                "creatinine_latest_24h": float(creatinine_latest),
                "bun_latest_24h": float(bun_latest),
                "wbc_max_24h": float(wbc_max),
                "platelets_min_24h": float(platelets_min),
                "bicarbonate_min_24h": float(bicarbonate_min),
            }
        ]
    )

    probability = float(
        pipeline.predict_proba(input_data)[0, 1]
    )

    category = risk_category(probability)

    st.divider()
    st.subheader("Model Output")

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Predicted Mortality Risk",
        f"{probability * 100:.2f}%",
    )

    result_col2.metric(
        "Demonstration Category",
        category,
    )

    result_col3.metric(
        "High-Risk Threshold",
        f"{HIGH_THRESHOLD:.2f}",
    )

    if category == "High Risk":
        st.error(
            "The model assigns these simulated inputs to the "
            "high-risk demonstration category."
        )
    elif category == "Medium Risk":
        st.warning(
            "The model assigns these simulated inputs to the "
            "medium-risk demonstration category."
        )
    else:
        st.success(
            "The model assigns these simulated inputs to the "
            "low-risk demonstration category."
        )

    st.caption(
        "Categories: below 0.21 = Low, 0.21–0.38 = Medium, "
        "0.38 or above = High. These thresholds are exploratory "
        "and have not been clinically validated."
    )

    st.subheader("Entered Clinical Features")

    st.dataframe(
        input_data.T.rename(columns={0: "Value"}),
        use_container_width=True,
    )

    st.subheader("SHAP Model Explanation")

    st.caption(
        "Positive SHAP values increase the model output, while "
        "negative values decrease it. SHAP explains model behavior "
        "and does not establish clinical causation."
    )

    shap_table = create_shap_table(
        pipeline,
        input_data,
    ).head(12)

    st.dataframe(
        shap_table[
            [
                "Feature",
                "Input Value",
                "SHAP Value",
                "Effect",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "This prediction is generated from a small demonstration "
        "dataset and must not be interpreted as a clinical risk "
        "assessment."
    )


if __name__ == "__main__":
    main()
