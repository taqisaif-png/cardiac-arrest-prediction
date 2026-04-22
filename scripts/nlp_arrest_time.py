"""
NLP extraction of cardiac arrest timing from clinical notes.

Two-stage approach:
  1. Regex around arrest-keyword sentences (fast, ~54/275 notes resolved)
  2. Stanza NER TIME/DATE entity extraction in the same windows (fallback)

Outputs a CSV with hadm_id, extracted time, and 15-min pre-arrest cutoff.
"""

import re
from datetime import timedelta

import dateparser
import duckdb
import pandas as pd
import stanza

DATA_DIR   = "/Users/taqi/Documents/umich/courses/WINTER 2026/LHS 712/final project"
NOTES_PATH = f"{DATA_DIR}/arrest_notes.csv"
LAB_PATH   = "/Volumes/storage/FINAL_PRJ_LHS/DATA/mimiciv/3.1/hosp/labevents.csv"
OUT_PATH   = f"{DATA_DIR}/nlp_v2_cutoffs.csv"

ARREST_KEYWORDS = {"arrest", "cpr", "code blue", "pulseless", "asystole", "rosc", "arrested"}
TIME_PATTERNS   = [
    r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\b",
    r"\b\d{1,2}:\d{2}\b",
]
WINDOW_LINES = 3  # lines before/after keyword sentence to search


def _match_time_in_window(window: str) -> str | None:
    for pattern in TIME_PATTERNS:
        matches = re.findall(pattern, window)
        if matches:
            return matches[0]
    return None


def extract_time_regex(text: str) -> str | None:
    sentences = text.split("\n")
    for i, sent in enumerate(sentences):
        if any(kw in sent.lower() for kw in ARREST_KEYWORDS):
            start = max(0, i - WINDOW_LINES)
            end   = min(len(sentences), i + WINDOW_LINES + 1)
            window = " ".join(sentences[start:end])
            result = _match_time_in_window(window)
            if result:
                return result
    return None


def extract_time_stanza(text: str, nlp) -> str | None:
    sentences = text[:15_000].split("\n")
    for i, sent in enumerate(sentences):
        if any(kw in sent.lower() for kw in ARREST_KEYWORDS):
            start  = max(0, i - WINDOW_LINES)
            end    = min(len(sentences), i + WINDOW_LINES + 1)
            window = " ".join(sentences[start:end])
            doc    = nlp(window)
            for ent in doc.entities:
                if ent.type in {"TIME", "DATE"}:
                    return ent.text
    return None


def load_lab_times(hadm_ids: list[int]) -> pd.DataFrame:
    ids = ",".join(map(str, hadm_ids))
    df  = duckdb.query(
        f"SELECT hadm_id, charttime FROM read_csv_auto('{LAB_PATH}') WHERE hadm_id IN ({ids})"
    ).df()
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def resolve_full_datetime(hadm_id: int, time_str: str, lab_times: pd.DataFrame):
    if pd.isna(time_str):
        return None
    patient_labs = lab_times[lab_times["hadm_id"] == hadm_id]
    if patient_labs.empty:
        return None
    for _, row in patient_labs.iterrows():
        parsed = dateparser.parse(f"{row['charttime'].date()} {time_str.strip()}")
        if parsed:
            return parsed
    return None


def run(use_stanza_fallback: bool = False) -> pd.DataFrame:
    notes = pd.read_csv(NOTES_PATH)
    print(f"Notes loaded: {notes.shape}")

    notes["nlp_time"] = notes["text"].apply(extract_time_regex)
    print(f"Regex extracted: {notes['nlp_time'].notna().sum()} / {len(notes)}")

    if use_stanza_fallback:
        stanza.download("en")
        nlp = stanza.Pipeline("en", processors="tokenize,ner", device="cpu")
        missing = notes["nlp_time"].isna()
        notes.loc[missing, "nlp_time"] = notes.loc[missing, "text"].apply(
            lambda t: extract_time_stanza(t, nlp)
        )
        print(f"After Stanza fallback: {notes['nlp_time'].notna().sum()} / {len(notes)}")

    lab_times = load_lab_times(notes["hadm_id"].tolist())

    notes["full_datetime"] = notes.apply(
        lambda r: resolve_full_datetime(r["hadm_id"], r["nlp_time"], lab_times), axis=1
    )
    print(f"Full datetime resolved: {notes['full_datetime'].notna().sum()}")

    cutoffs = notes[notes["full_datetime"].notna()][["hadm_id", "full_datetime"]].copy()
    cutoffs["cutoff_time"] = cutoffs["full_datetime"] - timedelta(minutes=15)
    cutoffs["source"]      = "nlp_regex"

    cutoffs.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(cutoffs)} cutoffs → {OUT_PATH}")
    return cutoffs


if __name__ == "__main__":
    run(use_stanza_fallback=False)
