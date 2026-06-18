from pathlib import Path

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

SUMMARY_PATH = Path(
    "outputs/metrics/model_comparison_cv.csv"
)

FOLD_RESULTS_PATH = Path(
    "outputs/metrics/model_comparison_cv_folds.csv"
)


BASE_FEATURES = [
    "gender",
    "anchor_age",
    "admission_type",
    "admission_location",
    "insurance",
    "race",
    "first_careunit",
]

VITAL_NAMES = [
    "heart_rate",
    "resp_rate",
    "spo2",
    "sbp",
    "dbp",
    "map",
    "temperature",
]

LAB_NAMES = [
    "lactate",
    "creatinine",
    "bun",
    "glucose",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "hemoglobin",
    "platelets",
    "wbc",
    "inr",
    "pt",
    "ptt",
]


def select_features(df, names):
    return [
        column
        for column in df.columns
        if column.endswith("_24h")
        and any(column.startswith(f"{name}_") for name in names)
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
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
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
    y = df[target].astype(int)

    vital_features = select_features(df, VITAL_NAMES)
    lab_features = select_features(df, LAB_NAMES)

    model_features = {
        "Baseline": BASE_FEATURES,
        "Vitals only": BASE_FEATURES + vital_features,
        "Vitals + labs": BASE_FEATURES + vital_features + lab_features,
    }

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

    summaries = []
    fold_results = []

    for model_name, feature_columns in model_features.items():
        X = df[feature_columns].copy()
        pipeline = build_pipeline(X, y)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=-1,
        )

        summary = {
            "model": model_name,
            "feature_count": len(feature_columns),
        }

        for metric in scoring:
            values = scores[f"test_{metric}"]

            summary[f"{metric}_mean"] = values.mean()
            summary[f"{metric}_std"] = values.std()

            for fold_number, value in enumerate(values, start=1):
                fold_results.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "metric": metric,
                        "value": value,
                    }
                )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    fold_df = pd.DataFrame(fold_results)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    fold_df.to_csv(FOLD_RESULTS_PATH, index=False)

    print("\nCross-validation comparison:\n")

    columns_to_show = [
        "model",
        "feature_count",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
        "recall_mean",
        "f1_mean",
    ]

    print(
        summary_df[columns_to_show]
        .sort_values("auprc_mean", ascending=False)
        .to_string(index=False)
    )

    print("\nSaved summary to:", SUMMARY_PATH)
    print("Saved fold results to:", FOLD_RESULTS_PATH)


if __name__ == "__main__":
    main()
