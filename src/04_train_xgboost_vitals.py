from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/icu_cohort_with_vitals_24h.csv")
MODEL_DIR = Path("outputs/models")
METRICS_DIR = Path("outputs/metrics")


def main():
    print("=" * 80)
    print("XGBoost Model with 24-hour ICU Vitals")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {DATA_PATH}. Run src/03_extract_vitals_features.py first."
        )

    df = pd.read_csv(DATA_PATH)

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

    vital_features = [
        col for col in df.columns
        if col.endswith("_24h")
    ]

    feature_cols = base_features + vital_features

    model_df = df[feature_cols + [target_col]].copy()
    model_df = model_df.dropna(subset=[target_col])

    print(f"\nLoaded dataset shape: {df.shape}")
    print(f"Model dataset shape: {model_df.shape}")
    print(f"Number of vitals features: {len(vital_features)}")

    print("\nTarget distribution:")
    print(model_df[target_col].value_counts())
    print(model_df[target_col].value_counts(normalize=True) * 100)

    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    X[categorical_cols] = X[categorical_cols].fillna("Unknown")

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    X_encoded = pd.get_dummies(X, drop_first=True)

    print(f"\nEncoded feature matrix shape: {X_encoded.shape}")

    if y.nunique() < 2:
        raise ValueError("Target has only one class. Cannot train model.")

    class_counts = y.value_counts()
    stratify_arg = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_arg,
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "num_features": int(X_encoded.shape[1]),
        "target_counts": y.value_counts().to_dict(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    if y_test.nunique() == 2:
        auroc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)

        print(f"\nAUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        metrics["auroc"] = float(auroc)
        metrics["auprc"] = float(auprc)
    else:
        print("\nAUROC/AUPRC skipped because test set has only one class.")
        metrics["auroc"] = None
        metrics["auprc"] = None

    importance_df = pd.DataFrame(
        {
            "feature": X_encoded.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 20 Feature Importances:")
    print(importance_df.head(20))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "xgboost_vitals_24h_mortality.joblib"
    metrics_path = METRICS_DIR / "xgboost_vitals_24h_metrics.json"
    importance_path = METRICS_DIR / "xgboost_vitals_24h_feature_importance.csv"

    joblib.dump(
        {
            "model": model,
            "feature_columns": X_encoded.columns.tolist(),
        },
        model_path,
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    importance_df.to_csv(importance_path, index=False)

    print("\nSaved model to:", model_path)
    print("Saved metrics to:", metrics_path)
    print("Saved feature importances to:", importance_path)

    print("\nVitals-based XGBoost training completed successfully.")


if __name__ == "__main__":
    main()
