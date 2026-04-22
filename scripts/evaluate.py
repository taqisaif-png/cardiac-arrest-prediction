"""
Feature-group ablation study for cardiac arrest prediction.

Trains a separate XGBoost model for each feature subset and reports
AUC + recall on MIMIC-IV internal test and eICU external validation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier

from train import RANDOM_STATE, NEG_RATIO, THRESHOLD, XGB_PARAMS, downsample


def define_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    vitals = [c for c in feature_cols if any(c.startswith(p) for p in
                                              ["hr_", "sbp_", "dbp_", "rr_", "spo2_"])]
    labs   = [c for c in feature_cols if any(c.startswith(p) for p in
                                              ["mean_", "max_", "min_"])]
    embeds = [c for c in feature_cols if c.startswith("cc_")]

    return {
        "vitals_only":        vitals,
        "labs_only":          labs,
        "embeddings_only":    embeds,
        "age_gender":         [c for c in ["age", "gender"] if c in feature_cols],
        "vitals_labs":        vitals + labs,
        "vitals_embeddings":  vitals + embeds,
        "labs_embeddings":    labs + embeds,
        "all_features":       feature_cols,
    }


def eval_group(
    group_name: str,
    cols: list[str],
    mimic_full: pd.DataFrame,
    eicu_full: pd.DataFrame,
    label_col: str = "label",
) -> dict:
    balanced = downsample(mimic_full, label_col)
    X = balanced[cols].fillna(balanced[cols].median())
    y = balanced[label_col]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X_tr, y_tr)

    y_prob_mimic = clf.predict_proba(X_te)[:, 1]
    auc_mimic    = roc_auc_score(y_te, y_prob_mimic)
    recall_mimic = classification_report(
        y_te, (y_prob_mimic >= THRESHOLD).astype(int), output_dict=True
    )["1"]["recall"]

    X_eicu      = eicu_full[cols].fillna(eicu_full[cols].median())
    y_eicu      = eicu_full[label_col]
    y_prob_eicu = clf.predict_proba(X_eicu)[:, 1]
    auc_eicu    = roc_auc_score(y_eicu, y_prob_eicu)
    recall_eicu = classification_report(
        y_eicu, (y_prob_eicu >= THRESHOLD).astype(int), output_dict=True
    )["1"]["recall"]

    print(f"{group_name:25s}  n={len(cols):4d}  "
          f"MIMIC AUC={auc_mimic:.3f} recall={recall_mimic:.2f}  "
          f"eICU AUC={auc_eicu:.3f} recall={recall_eicu:.2f}")

    return {
        "n_features":   len(cols),
        "mimic_auc":    auc_mimic,
        "mimic_recall": recall_mimic,
        "eicu_auc":     auc_eicu,
        "eicu_recall":  recall_eicu,
    }


def run_ablation(
    mimic_full: pd.DataFrame,
    eicu_full: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    groups = define_feature_groups(feature_cols)
    results = {}

    for name, cols in groups.items():
        eicu_cols = [c for c in cols if c in eicu_full.columns]
        for c in cols:
            if c not in eicu_full.columns:
                eicu_full[c] = np.nan
        results[name] = eval_group(name, cols, mimic_full, eicu_full)

    return pd.DataFrame(results).T.sort_values("mimic_auc", ascending=False)


if __name__ == "__main__":
    from embed_text import embed_eicu, embed_mimic
    from prepare_features import EXCLUDE_COLS, load_eicu, load_mimic
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    mimic = embed_mimic(load_mimic(), model)
    eicu  = embed_eicu(load_eicu(), model)

    feature_cols = [c for c in mimic.columns if c not in EXCLUDE_COLS]

    print("\n=== Feature Ablation ===\n")
    results = run_ablation(mimic, eicu, feature_cols)
    print("\n", results.to_string())
