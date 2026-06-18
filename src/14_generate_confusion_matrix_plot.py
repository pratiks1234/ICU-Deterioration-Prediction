from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


INPUT_PATH = Path(
    "outputs/metrics/reduced_clinical_oof_predictions_tuned.csv"
)

OUTPUT_PATH = Path(
    "docs/images/confusion_matrix_threshold_038.png"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    y_true = df["hospital_expire_flag"].astype(int)
    y_pred = df["predicted_class_tuned"].astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Survived",
            "Hospital mortality",
        ],
    )

    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        "Out-of-Fold Confusion Matrix\n"
        "Reduced Clinical Model — Threshold 0.38"
    )

    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Confusion matrix:")
    print(matrix)
    print("Saved plot:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
