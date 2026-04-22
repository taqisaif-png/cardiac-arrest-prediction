"""
XGBoost training for cardiac arrest prediction.

Training strategy:
  - 10:1 downsampling of negatives (preserves all positives)
  - Decision threshold tuned to 0.3 for higher recall
  - scale_pos_weight=10 for additional class-imbalance correction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier

THRESHOLD      = 0.3
NEG_RATIO      = 10    # negatives kept per positive
RANDOM_STATE   = 42

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=1.0,
    scale_pos_weight=NEG_RATIO,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="auc",
)


def downsample(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    neg = df[df[label_col] == 0]
    pos = df[df[label_col] == 1]
    neg_down = resample(neg, n_samples=len(pos) * NEG_RATIO, random_state=RANDOM_STATE)
    return pd.concat([neg_down, pos]).reset_index(drop=True)


def print_metrics(y_true, y_prob, dataset_name: str) -> dict:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, output_dict=True)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    print(f"\n{dataset_name}")
    print(classification_report(y_true, y_pred))
    print(f"AUC-ROC: {auc:.3f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {"auc": auc, "recall": report["1"]["recall"], "precision": report["1"]["precision"]}


def train(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def run_pipeline(
    mimic_full: pd.DataFrame,
    eicu_full: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
) -> XGBClassifier:
    balanced = downsample(mimic_full, label_col)
    X = balanced[feature_cols].fillna(balanced[feature_cols].median())
    y = balanced[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = train(X_train, y_train)

    y_prob_test = clf.predict_proba(X_test)[:, 1]
    print_metrics(y_test, y_prob_test, "MIMIC-IV internal test set")

    X_eicu = eicu_full[feature_cols].fillna(eicu_full[feature_cols].median())
    y_eicu = eicu_full[label_col]
    y_prob_eicu = clf.predict_proba(X_eicu)[:, 1]
    print_metrics(y_eicu, y_prob_eicu, "eICU external validation")

    return clf


if __name__ == "__main__":
    from embed_text import embed_eicu, embed_mimic
    from prepare_features import EXCLUDE_COLS, load_eicu, load_mimic
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    mimic = embed_mimic(load_mimic(), model)
    eicu  = embed_eicu(load_eicu(), model)

    feature_cols = [c for c in mimic.columns if c not in EXCLUDE_COLS]
    missing = [c for c in feature_cols if c not in eicu.columns]
    for c in missing:
        eicu[c] = np.nan

    clf = run_pipeline(mimic, eicu, feature_cols)
