from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)


INPUT_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_predictions_tuned.csv"
)

OUTPUT_PATH = Path(
    "docs/images/precision_recall_curve.png"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    y_true = df["hospital_expire_flag"].astype(int)
    probabilities = df["predicted_risk"].astype(float)

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    average_precision = average_precision_score(
        y_true,
        probabilities,
    )

    mortality_rate = y_true.mean()

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.plot(
        recall,
        precision,
        label=f"Reduced clinical model (AUPRC = {average_precision:.3f})",
    )

    axis.axhline(
        mortality_rate,
        linestyle="--",
        label=f"Mortality prevalence = {mortality_rate:.3f}",
    )

    axis.set_title(
        "Out-of-Fold Precision–Recall Curve"
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"OOF AUPRC: {average_precision:.4f}")
    print(f"Mortality prevalence: {mortality_rate:.4f}")
    print("Saved plot:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
