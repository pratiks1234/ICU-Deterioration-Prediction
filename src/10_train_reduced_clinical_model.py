from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = Path(
    "data/processed/icu_cohort_with_vitals_labs_24h.csv"
)

MODEL_PATH = Path(
    "outputs/models/xgboost_reduced_clinical_24h.joblib"
)

METRICS_PATH = Path(
    "outputs/metrics/xgboost_reduced_clinical_cv_metrics.json"
)

IMPORTANCE_PATH = Path(
    "outputs/metrics/xgboost_reduced_clinical_feature_importance.csv"
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


def build_pipeline(X, y):
    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_columns = [
        column
        for column in X.columns
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

    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())

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

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    model_df = df[FEATURE_COLUMNS + [target]].dropna(
        subset=[target]
    ).copy()

    X = model_df[FEATURE_COLUMNS]
    y = model_df[target].astype(int)

    pipeline = build_pipeline(X, y)

    cross_validation = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42,
    )

    scoring = {
        "auroc": "roc_auc",
        "auprc": "average_precision",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
    }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cross_validation,
        scoring=scoring,
        n_jobs=-1,
    )

    metrics = {
        "rows": int(len(model_df)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "features": FEATURE_COLUMNS,
    }

    print("=" * 70)
    print("Reduced Clinical Model - Cross-Validation Results")
    print("=" * 70)

    for metric in scoring:
        values = scores[f"test_{metric}"]

        mean_value = float(values.mean())
        std_value = float(values.std())

        metrics[f"{metric}_mean"] = mean_value
        metrics[f"{metric}_std"] = std_value

        print(
            f"{metric.upper():10s}: "
            f"{mean_value:.4f} ± {std_value:.4f}"
        )

    pipeline.fit(X, y)

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importances = pipeline.named_steps[
        "classifier"
    ].feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 20 features:")
    print(importance_df.head(20).to_string(index=False))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved feature importance:", IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
