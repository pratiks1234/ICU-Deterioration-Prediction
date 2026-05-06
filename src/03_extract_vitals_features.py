from pathlib import Path
import pandas as pd
import numpy as np


RAW = Path("data/raw")
PROCESSED = Path("data/processed")

COHORT_PATH = PROCESSED / "icu_cohort_demo.csv"
CHARTEVENTS_PATH = RAW / "icu" / "chartevents.csv.gz"
D_ITEMS_PATH = RAW / "icu" / "d_items.csv.gz"

OUTPUT_PATH = PROCESSED / "icu_cohort_with_vitals_24h.csv"


# Common MIMIC-IV itemids for ICU vitals
VITAL_ITEMIDS = {
    220045: "heart_rate",

    220210: "resp_rate",

    220277: "spo2",

    220179: "sbp",   # non-invasive systolic BP
    220050: "sbp",   # arterial systolic BP

    220180: "dbp",   # non-invasive diastolic BP
    220051: "dbp",   # arterial diastolic BP

    220181: "map",   # non-invasive mean BP
    220052: "map",   # arterial mean BP

    223761: "temperature",  # Fahrenheit
    223762: "temperature",  # Celsius
}


def clean_vitals(df):
    """Apply basic clinical range cleaning."""

    # Convert Fahrenheit temperature to Celsius when needed
    temp_mask = df["vital_name"] == "temperature"

    # If temperature value looks like Fahrenheit, convert to Celsius
    fahrenheit_mask = temp_mask & (df["valuenum"] > 70)
    df.loc[fahrenheit_mask, "valuenum"] = (
        df.loc[fahrenheit_mask, "valuenum"] - 32
    ) * 5 / 9

    ranges = {
        "heart_rate": (20, 250),
        "resp_rate": (1, 80),
        "spo2": (40, 100),
        "sbp": (40, 300),
        "dbp": (20, 200),
        "map": (20, 250),
        "temperature": (25, 45),
    }

    cleaned_parts = []

    for vital, (low, high) in ranges.items():
        part = df[df["vital_name"] == vital].copy()
        part = part[(part["valuenum"] >= low) & (part["valuenum"] <= high)]
        cleaned_parts.append(part)

    return pd.concat(cleaned_parts, ignore_index=True)


def main():
    print("=" * 80)
    print("Extracting 24-hour ICU Vitals Features")
    print("=" * 80)

    if not COHORT_PATH.exists():
        raise FileNotFoundError(f"Missing cohort file: {COHORT_PATH}")

    cohort = pd.read_csv(COHORT_PATH)
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")

    print(f"\nLoaded ICU cohort: {cohort.shape}")

    print("\nLoading d_items...")
    d_items = pd.read_csv(D_ITEMS_PATH)
    print(f"d_items shape: {d_items.shape}")

    print("\nSelected vital itemids:")
    vital_dict = d_items[d_items["itemid"].isin(VITAL_ITEMIDS.keys())][
        ["itemid", "label"]
    ].copy()
    vital_dict["vital_name"] = vital_dict["itemid"].map(VITAL_ITEMIDS)
    print(vital_dict.sort_values("vital_name"))

    print("\nLoading chartevents selected columns...")
    usecols = ["stay_id", "charttime", "itemid", "valuenum", "valueuom"]

    chartevents = pd.read_csv(
        CHARTEVENTS_PATH,
        usecols=usecols,
        low_memory=False,
    )

    print(f"Raw chartevents shape: {chartevents.shape}")

    # Keep only selected vitals
    vitals = chartevents[chartevents["itemid"].isin(VITAL_ITEMIDS.keys())].copy()
    vitals = vitals.dropna(subset=["valuenum"])

    vitals["charttime"] = pd.to_datetime(vitals["charttime"], errors="coerce")
    vitals["vital_name"] = vitals["itemid"].map(VITAL_ITEMIDS)

    print(f"Filtered vitals shape: {vitals.shape}")
    print("\nVitals count before cleaning:")
    print(vitals["vital_name"].value_counts())

    vitals = clean_vitals(vitals)

    print("\nVitals count after cleaning:")
    print(vitals["vital_name"].value_counts())

    # Merge ICU intime so we can keep only first 24 hours
    vitals = vitals.merge(
        cohort[["stay_id", "intime"]],
        on="stay_id",
        how="inner",
    )

    vitals["hours_from_icu_admit"] = (
        vitals["charttime"] - vitals["intime"]
    ).dt.total_seconds() / 3600

    # First 24 hours after ICU admission
    vitals_24h = vitals[
        (vitals["hours_from_icu_admit"] >= 0)
        & (vitals["hours_from_icu_admit"] <= 24)
    ].copy()

    print(f"\nVitals in first 24h shape: {vitals_24h.shape}")

    print("\nCoverage by vital in first 24h:")
    print(vitals_24h.groupby("vital_name")["stay_id"].nunique().sort_values(ascending=False))

    # Summary statistics
    agg = (
        vitals_24h
        .groupby(["stay_id", "vital_name"])["valuenum"]
        .agg(["mean", "min", "max", "std", "count"])
        .reset_index()
    )

    feature_frames = []

    for stat in ["mean", "min", "max", "std", "count"]:
        pivot = agg.pivot(index="stay_id", columns="vital_name", values=stat)
        pivot.columns = [f"{col}_{stat}_24h" for col in pivot.columns]
        feature_frames.append(pivot)

    # Latest value in first 24h
    latest = (
        vitals_24h
        .sort_values(["stay_id", "vital_name", "charttime"])
        .groupby(["stay_id", "vital_name"])
        .tail(1)
        .pivot(index="stay_id", columns="vital_name", values="valuenum")
    )
    latest.columns = [f"{col}_latest_24h" for col in latest.columns]
    feature_frames.append(latest)

    vitals_features = pd.concat(feature_frames, axis=1).reset_index()

    print(f"\nVitals feature table shape: {vitals_features.shape}")
    print("\nVitals feature preview:")
    print(vitals_features.head())

    # Add missing flags
    vital_names = sorted(set(VITAL_ITEMIDS.values()))

    for vital in vital_names:
        count_col = f"{vital}_count_24h"
        missing_col = f"{vital}_missing_24h"

        if count_col in vitals_features.columns:
            vitals_features[missing_col] = vitals_features[count_col].isna().astype(int)
        else:
            vitals_features[missing_col] = 1

    # Merge with cohort
    final_df = cohort.merge(vitals_features, on="stay_id", how="left")

    print(f"\nFinal cohort with vitals shape: {final_df.shape}")

    print("\nNew columns added:")
    new_cols = [c for c in final_df.columns if c not in cohort.columns]
    print(new_cols)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved final vitals dataset to: {OUTPUT_PATH}")
    print("Vitals feature extraction completed successfully.")


if __name__ == "__main__":
    main()
