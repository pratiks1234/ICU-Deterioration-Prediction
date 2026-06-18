from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = Path(
    "data/processed/icu_cohort_with_vitals_labs_24h.csv"
)

PREDICTIONS_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_predictions.csv"
)

METRICS_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_metrics.json"
)

FINAL_MODEL_PATH = Path(
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


def build_pipeline(X_train, y_train):
    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_columns = [
        column
        for column in X_train.columns
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
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.2,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    target = "hospital_expire_flag"

    required_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        target,
    ] + FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    model_df = df[required_columns].dropna(
        subset=[target]
    ).reset_index(drop=True)

    X = model_df[FEATURE_COLUMNS]
    y = model_df[target].astype(int)

    oof_probabilities = np.zeros(len(model_df))
    oof_folds = np.zeros(len(model_df), dtype=int)

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    for fold_number, (train_indices, test_indices) in enumerate(
        cross_validation.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]

        pipeline = build_pipeline(X_train, y_train)
        pipeline.fit(X_train, y_train)

        fold_probabilities = pipeline.predict_proba(X_test)[:, 1]

        oof_probabilities[test_indices] = fold_probabilities
        oof_folds[test_indices] = fold_number

        print(
            f"Fold {fold_number}: "
            f"train={len(train_indices)}, "
            f"test={len(test_indices)}"
        )

    oof_predictions = (oof_probabilities >= 0.50).astype(int)

    auroc = roc_auc_score(y, oof_probabilities)
    auprc = average_precision_score(y, oof_probabilities)
    matrix = confusion_matrix(y, oof_predictions)

    report_text = classification_report(
        y,
        oof_predictions,
        zero_division=0,
    )

    report_dict = classification_report(
        y,
        oof_predictions,
        zero_division=0,
        output_dict=True,
    )

    print("\nOut-of-fold classification report:")
    print(report_text)

    print("\nConfusion matrix:")
    print(matrix)

    print(f"\nOOF AUROC: {auroc:.4f}")
    print(f"OOF AUPRC: {auprc:.4f}")

    results = model_df[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            target,
        ]
    ].copy()

    results["fold"] = oof_folds
    results["predicted_risk"] = oof_probabilities
    results["predicted_class_050"] = oof_predictions

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(PREDICTIONS_PATH, index=False)

    metrics = {
        "rows": int(len(model_df)),
        "folds": 5,
        "feature_count": len(FEATURE_COLUMNS),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    final_pipeline = build_pipeline(X, y)
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, FINAL_MODEL_PATH)

    print("\nSaved OOF predictions:", PREDICTIONS_PATH)
    print("Saved OOF metrics:", METRICS_PATH)
    print("Saved final model:", FINAL_MODEL_PATH)


if __name__ == "__main__":
    main()
