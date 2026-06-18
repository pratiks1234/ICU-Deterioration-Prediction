from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


INPUT_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_predictions.csv"
)

RESULTS_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_threshold_results.csv"
)

SUMMARY_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_threshold_summary.json"
)

PREDICTIONS_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_predictions_tuned.csv"
)


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main():
    df = pd.read_csv(INPUT_PATH)

    y_true = df["hospital_expire_flag"].astype(int)
    probabilities = df["predicted_risk"].astype(float)

    thresholds = np.arange(0.05, 0.96, 0.01)

    results = [
        calculate_metrics(y_true, probabilities, threshold)
        for threshold in thresholds
    ]

    results_df = pd.DataFrame(results)

    best_f1 = results_df.loc[
        results_df["f1_score"].idxmax()
    ]

    high_recall_candidates = results_df[
        results_df["recall"] >= 0.70
    ]

    if not high_recall_candidates.empty:
        high_recall = high_recall_candidates.sort_values(
            ["precision", "f1_score"],
            ascending=False,
        ).iloc[0]
    else:
        high_recall = results_df.loc[
            results_df["recall"].idxmax()
        ]

    recommended_threshold = float(best_f1["threshold"])

    print("=" * 70)
    print("Out-of-Fold Threshold Tuning")
    print("=" * 70)

    print("\nBest threshold by F1-score:")
    print(best_f1.to_string())

    print("\nHigh-recall threshold:")
    print(high_recall.to_string())

    print("\nComparison with threshold 0.50:")
    default_row = results_df.iloc[
        (results_df["threshold"] - 0.50).abs().argsort()[:1]
    ]
    print(default_row.to_string(index=False))

    df["predicted_class_tuned"] = (
        df["predicted_risk"] >= recommended_threshold
    ).astype(int)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(RESULTS_PATH, index=False)
    df.to_csv(PREDICTIONS_PATH, index=False)

    summary = {
        "recommended_threshold": recommended_threshold,
        "selection_method": "highest_f1_score",
        "best_f1_result": {
            key: float(value)
            for key, value in best_f1.items()
        },
        "high_recall_result": {
            key: float(value)
            for key, value in high_recall.items()
        },
    }

    with open(SUMMARY_PATH, "w") as file:
        json.dump(summary, file, indent=4)

    print("\nRecommended threshold:", recommended_threshold)
    print("Saved threshold results:", RESULTS_PATH)
    print("Saved threshold summary:", SUMMARY_PATH)
    print("Saved tuned predictions:", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
