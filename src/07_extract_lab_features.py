from pathlib import Path
import pandas as pd


RAW = Path("data/raw")
PROCESSED = Path("data/processed")

INPUT_PATH = PROCESSED / "icu_cohort_with_vitals_24h.csv"
LABEVENTS_PATH = RAW / "hosp" / "labevents.csv.gz"
D_LABITEMS_PATH = RAW / "hosp" / "d_labitems.csv.gz"
OUTPUT_PATH = PROCESSED / "icu_cohort_with_vitals_labs_24h.csv"


LAB_ITEMIDS = {
    50813: "lactate",
    50912: "creatinine",
    51006: "bun",
    50931: "glucose",
    50809: "glucose",
    50983: "sodium",
    50971: "potassium",
    50902: "chloride",
    50882: "bicarbonate",
    51222: "hemoglobin",
    51265: "platelets",
    51301: "wbc",
    51300: "wbc",
    51237: "inr",
    51274: "pt",
    51275: "ptt",
}


LAB_RANGES = {
    "lactate": (0, 30),
    "creatinine": (0, 20),
    "bun": (0, 250),
    "glucose": (20, 1000),
    "sodium": (90, 180),
    "potassium": (1, 10),
    "chloride": (60, 140),
    "bicarbonate": (5, 60),
    "hemoglobin": (2, 25),
    "platelets": (1, 2000),
    "wbc": (0, 500),
    "inr": (0, 20),
    "pt": (0, 200),
    "ptt": (0, 300),
}


def clean_labs(labs):
    parts = []

    for lab_name, (low, high) in LAB_RANGES.items():
        part = labs[labs["lab_name"] == lab_name].copy()
        part = part[part["valuenum"].between(low, high)]
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def main():
    print("=" * 70)
    print("Extracting First-24-Hour Lab Features")
    print("=" * 70)

    cohort = pd.read_csv(INPUT_PATH)
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")

    print("Cohort shape:", cohort.shape)

    d_labitems = pd.read_csv(D_LABITEMS_PATH)

    selected_items = d_labitems[
        d_labitems["itemid"].isin(LAB_ITEMIDS)
    ][["itemid", "label"]].copy()

    selected_items["lab_name"] = selected_items["itemid"].map(LAB_ITEMIDS)

    print("\nSelected labs:")
    print(selected_items.sort_values("lab_name").to_string(index=False))

    labevents = pd.read_csv(
        LABEVENTS_PATH,
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        low_memory=False,
    )

    labs = labevents[labevents["itemid"].isin(LAB_ITEMIDS)].copy()
    labs = labs.dropna(subset=["valuenum", "charttime"])

    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    labs["lab_name"] = labs["itemid"].map(LAB_ITEMIDS)

    labs = clean_labs(labs)

    labs = labs.merge(
        cohort[["subject_id", "hadm_id", "stay_id", "intime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    labs["hours_from_icu_admit"] = (
        labs["charttime"] - labs["intime"]
    ).dt.total_seconds() / 3600

    labs_24h = labs[
        labs["hours_from_icu_admit"].between(0, 24)
    ].copy()

    print("\nLab rows in first 24 hours:", len(labs_24h))

    print("\nICU-stay coverage:")
    print(
        labs_24h.groupby("lab_name")["stay_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    grouped = (
        labs_24h.groupby(["stay_id", "lab_name"])["valuenum"]
        .agg(["mean", "min", "max", "std", "count"])
        .reset_index()
    )

    feature_tables = []

    for statistic in ["mean", "min", "max", "std", "count"]:
        table = grouped.pivot(
            index="stay_id",
            columns="lab_name",
            values=statistic,
        )
        table.columns = [
            f"{lab}_{statistic}_24h" for lab in table.columns
        ]
        feature_tables.append(table)

    latest = (
        labs_24h.sort_values(["stay_id", "lab_name", "charttime"])
        .groupby(["stay_id", "lab_name"])
        .tail(1)
        .pivot(index="stay_id", columns="lab_name", values="valuenum")
    )

    latest.columns = [
        f"{lab}_latest_24h" for lab in latest.columns
    ]
    feature_tables.append(latest)

    lab_features = pd.concat(feature_tables, axis=1).reset_index()

    final_df = cohort.merge(lab_features, on="stay_id", how="left")

    for lab_name in sorted(set(LAB_ITEMIDS.values())):
        count_column = f"{lab_name}_count_24h"
        missing_column = f"{lab_name}_missing_24h"

        if count_column not in final_df.columns:
            final_df[count_column] = 0

        final_df[count_column] = final_df[count_column].fillna(0)
        final_df[missing_column] = (
            final_df[count_column] == 0
        ).astype(int)

    final_df.to_csv(OUTPUT_PATH, index=False)

    new_columns = [
        column for column in final_df.columns
        if column not in cohort.columns
    ]

    print("\nFinal dataset shape:", final_df.shape)
    print("Lab columns added:", len(new_columns))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
