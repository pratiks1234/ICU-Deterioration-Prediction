from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


DATA_PATH = Path("data/processed/icu_cohort_with_vitals_24h.csv")
MODEL_PATH = Path("outputs/models/xgboost_vitals_24h_mortality.joblib")
OUTPUT_PATH = Path("outputs/metrics/threshold_tuning_results.csv")


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
    print("Threshold Tuning for XGBoost Vitals Model")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    saved_features = saved_model["feature_columns"]

    X, y = prepare_features(df)

    X = X.reindex(columns=saved_features, fill_value=0)

    class_counts = y.value_counts()
    stratify_arg = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_arg,
    )

    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nThreshold tuning results:")
    print(results_df)

    best_f1 = results_df.sort_values("f1_score", ascending=False).iloc[0]

    print("\nBest threshold by F1-score:")
    print(best_f1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved threshold tuning results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
