from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CV_RESULTS_PATH = Path(
    "outputs/metrics/model_comparison_cv.csv"
)

REDUCED_RESULTS_PATH = Path(
    "outputs/metrics/xgboost_reduced_clinical_cv_metrics.json"
)

OUTPUT_PATH = Path(
    "docs/images/model_comparison.png"
)


def main():
    comparison_df = pd.read_csv(CV_RESULTS_PATH)

    with open(REDUCED_RESULTS_PATH) as file:
        reduced_metrics = json.load(file)

    reduced_row = {
        "model": "Reduced clinical",
        "feature_count": reduced_metrics["feature_count"],
        "auroc_mean": reduced_metrics["auroc_mean"],
        "auroc_std": reduced_metrics["auroc_std"],
        "auprc_mean": reduced_metrics["auprc_mean"],
        "auprc_std": reduced_metrics["auprc_std"],
    }

    plot_df = pd.concat(
        [
            comparison_df[
                [
                    "model",
                    "feature_count",
                    "auroc_mean",
                    "auroc_std",
                    "auprc_mean",
                    "auprc_std",
                ]
            ],
            pd.DataFrame([reduced_row]),
        ],
        ignore_index=True,
    )

    model_labels = [
        f"{row['model']}\n({int(row['feature_count'])} features)"
        for _, row in plot_df.iterrows()
    ]

    positions = np.arange(len(plot_df))
    width = 0.35

    figure, axis = plt.subplots(figsize=(11, 6))

    auroc_bars = axis.bar(
        positions - width / 2,
        plot_df["auroc_mean"],
        width,
        yerr=plot_df["auroc_std"],
        capsize=5,
        label="AUROC",
    )

    auprc_bars = axis.bar(
        positions + width / 2,
        plot_df["auprc_mean"],
        width,
        yerr=plot_df["auprc_std"],
        capsize=5,
        label="AUPRC",
    )

    axis.set_title(
        "Repeated Cross-Validation Model Comparison"
    )
    axis.set_ylabel("Mean score")
    axis.set_xlabel("Model")
    axis.set_xticks(positions)
    axis.set_xticklabels(model_labels)
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    axis.bar_label(
        auroc_bars,
        labels=[
            f"{value:.3f}"
            for value in plot_df["auroc_mean"]
        ],
        padding=3,
    )

    axis.bar_label(
        auprc_bars,
        labels=[
            f"{value:.3f}"
            for value in plot_df["auprc_mean"]
        ],
        padding=3,
    )

    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print("Saved plot:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
