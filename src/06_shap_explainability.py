from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/icu_cohort_with_vitals_24h.csv")
MODEL_PATH = Path("outputs/models/xgboost_vitals_24h_mortality.joblib")
OUTPUT_DIR = Path("outputs/figures")
METRICS_DIR = Path("outputs/metrics")


def prepare_features(df):
    target_col = "hospital_expire_flag"

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

    model_df = df[feature_cols + [target_col]].copy()
    model_df = model_df.dropna(subset=[target_col])

    X = model_df[feature_cols].copy()
    y = model_df[target_col].astype(int)

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    X[categorical_cols] = X[categorical_cols].fillna("Unknown")

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y


def main():
    print("=" * 80)
    print("SHAP Explainability for XGBoost Vitals Model")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    saved_features = saved_model["feature_columns"]

    X, y = prepare_features(df)
    X = X.reindex(columns=saved_features, fill_value=0)

    class_counts = y.value_counts()
    stratify_arg = y if class_counts.min() >= 2 else None

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_arg,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    print("\nTop SHAP features:")
    print(shap_df.head(20))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    shap_csv_path = METRICS_DIR / "shap_global_importance.csv"
    shap_df.to_csv(shap_csv_path, index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    shap_plot_path = OUTPUT_DIR / "shap_summary_bar.png"
    plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    risk_scores = model.predict_proba(X_test)[:, 1]
    patient_idx = risk_scores.argmax()

    patient_explanation = pd.DataFrame(
        {
            "feature": X_test.columns,
            "feature_value": X_test.iloc[patient_idx].values,
            "shap_value": shap_values[patient_idx],
        }
    )

    patient_explanation["abs_shap"] = patient_explanation["shap_value"].abs()
    patient_explanation = patient_explanation.sort_values("abs_shap", ascending=False)

    patient_path = METRICS_DIR / "highest_risk_patient_shap.csv"
    patient_explanation.head(20).to_csv(patient_path, index=False)

    print(f"\nSaved global SHAP importance to: {shap_csv_path}")
    print(f"Saved SHAP summary plot to: {shap_plot_path}")
    print(f"Saved highest-risk patient explanation to: {patient_path}")

    print("\nHighest-risk patient predicted risk:", risk_scores[patient_idx])
    print("\nTop patient-level contributing features:")
    print(patient_explanation.head(10))


if __name__ == "__main__":
    main()
