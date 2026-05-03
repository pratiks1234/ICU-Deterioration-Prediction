from pathlib import Path
import pandas as pd


RAW = Path("data/raw")


def load_csv(path):
    """Load a compressed CSV file and print basic information."""
    print(f"\nLoading: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def main():
    print("=" * 80)
    print("MIMIC-IV Demo Data Check")
    print("=" * 80)

    patients_path = RAW / "hosp" / "patients.csv.gz"
    admissions_path = RAW / "hosp" / "admissions.csv.gz"
    icustays_path = RAW / "icu" / "icustays.csv.gz"

    patients = load_csv(patients_path)
    admissions = load_csv(admissions_path)
    icustays = load_csv(icustays_path)

    print("\n" + "=" * 80)
    print("Basic Table Preview")
    print("=" * 80)

    print("\nPatients:")
    print(patients.head())

    print("\nAdmissions:")
    print(admissions.head())

    print("\nICU Stays:")
    print(icustays.head())

    print("\n" + "=" * 80)
    print("Creating ICU Cohort")
    print("=" * 80)

    # Merge ICU stays with admissions to get hospital mortality label
    icu_cohort = icustays.merge(
        admissions[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "admission_type",
                "admission_location",
                "discharge_location",
                "insurance",
                "race",
                "hospital_expire_flag",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    # Merge patient demographics
    icu_cohort = icu_cohort.merge(
        patients[
            [
                "subject_id",
                "gender",
                "anchor_age",
                "anchor_year",
                "anchor_year_group",
                "dod",
            ]
        ],
        on="subject_id",
        how="left",
    )

    # Convert times
    time_cols = ["intime", "outtime", "admittime", "dischtime", "deathtime", "dod"]
    for col in time_cols:
        if col in icu_cohort.columns:
            icu_cohort[col] = pd.to_datetime(icu_cohort[col], errors="coerce")

    # Calculate ICU length of stay in hours
    icu_cohort["icu_los_hours"] = (
        icu_cohort["outtime"] - icu_cohort["intime"]
    ).dt.total_seconds() / 3600

    print(f"\nICU cohort shape: {icu_cohort.shape}")
    print("\nICU cohort columns:")
    print(icu_cohort.columns.tolist())

    print("\nICU cohort preview:")
    print(icu_cohort.head())

    print("\n" + "=" * 80)
    print("Target Label Distribution")
    print("=" * 80)

    print("\nRaw counts:")
    print(icu_cohort["hospital_expire_flag"].value_counts(dropna=False))

    print("\nPercentages:")
    print(icu_cohort["hospital_expire_flag"].value_counts(normalize=True, dropna=False) * 100)

    print("\n" + "=" * 80)
    print("Basic Cohort Summary")
    print("=" * 80)

    print(f"Unique patients: {icu_cohort['subject_id'].nunique()}")
    print(f"Unique hospital admissions: {icu_cohort['hadm_id'].nunique()}")
    print(f"Unique ICU stays: {icu_cohort['stay_id'].nunique()}")

    print("\nGender distribution:")
    print(icu_cohort["gender"].value_counts(dropna=False))

    print("\nAge summary:")
    print(icu_cohort["anchor_age"].describe())

    print("\nICU length of stay in hours:")
    print(icu_cohort["icu_los_hours"].describe())

    print("\nFirst care unit distribution:")
    print(icu_cohort["first_careunit"].value_counts(dropna=False))

    print("\n" + "=" * 80)
    print("Saving Processed ICU Cohort")
    print("=" * 80)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "icu_cohort_demo.csv"
    icu_cohort.to_csv(output_path, index=False)

    print(f"Saved ICU cohort to: {output_path}")
    print("\nData check completed successfully.")


if __name__ == "__main__":
    main()
