"""
Feature preparation for cardiac arrest prediction.

Loads MIMIC-IV and eICU datasets, applies clinical range validation,
encodes categorical variables, and saves cleaned versions ready for embedding + modeling.
"""

import numpy as np
import pandas as pd

DATA_DIR = "/Users/taqi/Documents/umich/courses/WINTER 2026/LHS 712/final project"

MIMIC_PATH = f"{DATA_DIR}/final_data_complete.csv"
EICU_PATH  = f"{DATA_DIR}/extra_data_1.csv"

EXCLUDE_COLS = [
    "ed_stay_id", "hadm_id", "subject_id",
    "ed_intime", "ed_outtime",
    "icu_stay_id", "icu_intime", "icu_outtime", "icu_los",
    "first_careunit", "label", "chiefcomplaint", "pain",
]

VITALS_RANGES = {
    "hr":   (0, 300),
    "sbp":  (0, 300),
    "dbp":  (0, 200),
    "rr":   (0, 60),
    "spo2": (50, 100),
}


def clean_pain(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    if val == "critical":
        return 11.0
    if val in {"unable", "uta", "ett", "n/a", "na", "none", ""}:
        return np.nan
    try:
        num = float(val)
        return num if 0 <= num <= 10 else np.nan
    except ValueError:
        return np.nan


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].map({"M": 1, "F": 0, "Male": 1, "Female": 0,
                                      "Unknown": np.nan, "Other": np.nan})
    transport_map = {"WALK IN": 0, "UNKNOWN": 1, "OTHER": 1, "AMBULANCE": 2, "HELICOPTER": 3}
    if "arrival_transport" in df.columns:
        df["arrival_transport"] = df["arrival_transport"].map(transport_map)
    return df


def validate_vitals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for prefix, (lo, hi) in VITALS_RANGES.items():
        cols = [c for c in df.columns if c.startswith(f"{prefix}_mean_")]
        for col in cols:
            df[col] = df[col].where(df[col].between(lo, hi), np.nan)

    # SBP must exceed DBP
    for i in range(1, 6):
        sbp_col, dbp_col = f"sbp_mean_{i}", f"dbp_mean_{i}"
        if sbp_col in df.columns and dbp_col in df.columns:
            bad = df[sbp_col] <= df[dbp_col]
            df.loc[bad, sbp_col] = np.nan
            df.loc[bad, dbp_col] = np.nan
    return df


def load_mimic(path: str = MIMIC_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pain_score"] = df["pain"].apply(clean_pain)
    df = encode_categoricals(df)
    print(f"MIMIC-IV: {df.shape}  |  arrests: {df['label'].sum()}")
    return df


def load_eicu(path: str = EICU_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["mean_pH"]       = df["mean_pH"].where(df["mean_pH"].between(6.5, 8.0), np.nan)
    df["min_pH"]        = df["min_pH"].where(df["min_pH"].between(6.5, 8.0), np.nan)
    df["mean_pO2"]      = df["mean_pO2"].where(df["mean_pO2"].between(0, 700), np.nan)
    df["min_pO2"]       = df["min_pO2"].where(df["min_pO2"].between(0, 700), np.nan)
    df["mean_Phosphate"] = df["mean_Phosphate"].where(df["mean_Phosphate"].between(0, 20), np.nan)

    df = validate_vitals(df)
    df = encode_categoricals(df)
    print(f"eICU: {df.shape}  |  arrests: {df['label'].sum()}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


if __name__ == "__main__":
    mimic = load_mimic()
    eicu  = load_eicu()
    print("\nMIMIC feature count:", len(get_feature_cols(mimic)))
    print("eICU  feature count:", len(get_feature_cols(eicu)))
