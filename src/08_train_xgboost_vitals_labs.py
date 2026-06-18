from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/icu_cohort_with_vitals_labs_24h.csv")
MODEL_PATH = Path("outputs/models/xgboost_vitals_labs_24h_mortality.joblib")
METRICS_PATH = Path("outputs/metrics/xgboost_vitals_labs_24h_metrics.json")
IMPORTANCE_PATH = Path(
    "outputs/metrics/xgboost_vitals_labs_24h_feature_importance.csv"
)
PREDICTIONS_PATH = Path(
    "outputs/metrics/xgboost_vitals_labs_24h_test_predictions.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)
    target = "hospital_expire_flag"

    base_features = [
        "gender",
        "anchor_age",
        "admission_type",
        "admission_location",
        "insurance",
        "race",
        "first_careunit",
    ]

    clinical_features = [
        column for column in df.columns
        if column.endswith("_24h")
    ]

    feature_columns = base_features + clinical_features

    model_df = df[
        ["subject_id", "hadm_id", "stay_id"] + feature_columns + [target]
    ].dropna(subset=[target]).copy()

    X = model_df[feature_columns]
    y = model_df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1
    )

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    auroc = roc_auc_score(y_test, probabilities)
    auprc = average_precision_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions)

    report_text = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True,
    )

    print("Dataset rows:", len(model_df))
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Input features:", len(feature_columns))

    print("\nClassification report:")
    print(report_text)

    print("\nConfusion matrix:")
    print(matrix)

    print(f"\nAUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    feature_importances = pipeline.named_steps[
        "classifier"
    ].feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importances,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 20 features:")
    print(importance_df.head(20).to_string(index=False))

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "input_features": int(len(feature_columns)),
        "target_counts": {
            str(label): int(count)
            for label, count in y.value_counts().items()
        },
        "auroc": float(auroc),
        "auprc": float(auprc),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    test_results = model_df.loc[
        X_test.index,
        ["subject_id", "hadm_id", "stay_id", target],
    ].copy()

    test_results["predicted_risk"] = probabilities
    test_results["predicted_class_050"] = predictions
    test_results.to_csv(PREDICTIONS_PATH, index=False)

    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved feature importance:", IMPORTANCE_PATH)
    print("Saved test predictions:", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
