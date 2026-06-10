from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/icu_cohort_demo.csv")
MODEL_DIR = Path("outputs/models")
METRICS_DIR = Path("outputs/metrics")


def main():
    print("=" * 80)
    print("Baseline XGBoost Model - ICU Mortality Prediction")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)

    target_col = "hospital_expire_flag"

    feature_cols = [
        "gender",
        "anchor_age",
        "admission_type",
        "admission_location",
        "insurance",
        "race",
        "first_careunit",
        "last_careunit",
    ]

    model_df = df[feature_cols + [target_col]].copy()
    model_df = model_df.dropna(subset=[target_col])

    print("\nTarget distribution:")
    print(model_df[target_col].value_counts())
    print(model_df[target_col].value_counts(normalize=True) * 100)

    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)

    X_encoded = pd.get_dummies(X, drop_first=True)

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
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "num_features": int(X_encoded.shape[1]),
        "target_counts": y.value_counts().to_dict(),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
    }

    if y_test.nunique() == 2:
        auroc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)

        print(f"\nAUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        metrics["auroc"] = float(auroc)
        metrics["auprc"] = float(auprc)

    importance_df = pd.DataFrame(
        {
            "feature": X_encoded.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop Feature Importances:")
    print(importance_df.head(15))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": X_encoded.columns.tolist(),
        },
        MODEL_DIR / "baseline_xgboost_mortality.joblib",
    )

    with open(METRICS_DIR / "baseline_xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    importance_df.to_csv(
        METRICS_DIR / "baseline_xgboost_feature_importance.csv",
        index=False,
    )

    print("\nBaseline training completed successfully.")


if __name__ == "__main__":
    main()
