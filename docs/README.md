# Cardiac Arrest Prediction from ED Presentations

Early prediction of in-hospital cardiac arrest using structured vitals, labs, and free text (MIMIC-IV chief complaints; eICU admission diagnoses) from MIMIC-IV (development) and eICU (external validation). Includes a label audit that separates true in-hospital arrests from patients who arrested before arrival.

---

## Background

Cardiac arrest (ICD-10: I46.x) is rare (~1.8% of ED-to-ICU transfers) but catastrophic. This project asks: **can we predict arrest before it happens, using only data available during the ED stay?**

We train on patients who transitioned from the ED directly to an ICU (within 1 hour), using triage vitals, serial ED vitals, laboratory values, and sentence-transformer embeddings of the chief complaint. The model is then validated zero-shot on eICU patients.

---

## Dataset

| Dataset | Patients | Arrests | Arrest rate |
|---------|----------|---------|-------------|
| MIMIC-IV v3.1 (train/test) | 19,065 | 339 | 1.8% |
| eICU (external validation) | 86,230 | 2,837 | 3.3% |

**Cohort criteria (MIMIC-IV):**
- Adult ED visit (age ≥ 18) with a subsequent ICU admission within 1 hour of ED discharge
- ED stay ≥ 2 hours
- At least one vital sign recorded during the ED stay

**Label:** ICD-10 codes I469, I468, I4601, I4609, or legacy 4275 on the associated hospitalization.

---

## Features

| Group | Features | Description |
|-------|----------|-------------|
| Demographics | `age`, `gender` | Age at admission, binary gender |
| Triage | `triage_hr/sbp/dbp/rr/spo2`, `acuity`, `pain_score`, `arrival_transport` | First ED measurements |
| Serial vitals | `{hr,sbp,dbp,rr,spo2}_mean_{1-5}` | Mean per quintile of ED stay |
| Labs | `mean/max/min_{Lactate,TroponinT,pH,pO2,...}` | 20 clinically relevant panels |
| Free text | `cc_0` … `cc_767` | 768-dim all-mpnet-base-v2 embedding of the MIMIC-IV chief complaint, or the eICU APACHE admission diagnosis (`admission_dx`) |

Total: ~835 features in the full model (57 structured + 768 embedding).

---

## Results

| Feature set | MIMIC-IV AUC | MIMIC-IV Recall | eICU AUC | eICU Recall |
|-------------|:---:|:---:|:---:|:---:|
| vitals only | 0.730 | 0.25 | 0.586 | 0.02 |
| labs only | 0.894 | 0.54 | 0.766 | 0.39 |
| embeddings only | 0.690 | 0.57 | 0.922 | 0.00 |
| vitals + labs | 0.922 | 0.56 | 0.801 | 0.26 |
| labs + embeddings | 0.905 | 0.49 | 0.951 | 0.60 |
| **all features** | **0.921** | **0.51** | **0.955** | **0.47** |

Decision threshold: 0.3 (tuned for recall on the positive class). AUC measures ranking, while recall at a fixed threshold depends on calibration, which does not transfer across datasets. That is why a feature set can have high eICU AUC and near-zero eICU recall.

### Interpreting the eICU results: label leakage

The eICU AUC (0.955) is higher than the internal MIMIC-IV AUC (0.921), which is a warning sign, not a success. Two sources of leakage inflate it:

1. **The label includes out-of-hospital arrests.** The ICD code is attached to the whole hospitalization, so patients who arrested *before* arriving (OHCA) are labeled positive, even though there is nothing left to predict.
2. **eICU's text field can contain the answer.** The eICU embedding uses the APACHE admission diagnosis (`admission_dx`), which is assigned at ICU admission and can directly state "cardiac arrest." The embeddings-only model reaching 0.922 AUC on eICU reflects this.

These results are kept for transparency, but they overstate true predictive performance.

---

## Label Audit and In-Hospital-Only Evaluation

To measure real prediction of in-hospital cardiac arrest (IHCA), the label itself was audited:

- Extracted arrest times from discharge notes with regex and Stanza NER (`scripts/nlp_arrest_time.py`).
- Hand-reviewed 104 notes in a Streamlit chart-review tool to separate out-of-hospital from in-hospital arrests.
- Trained a TF-IDF classifier on the reviewed notes to separate OHCA from IHCA at scale (CV AUC 0.909).

On the stricter IHCA-only test set, the model reaches **AUC 0.814**, compared with **~0.51** for the MEWS and NEWS early-warning scores. This is the headline result: lower than the leaky numbers above, but it measures what the model is meant to do.

---

## Project Structure

```
.
├── docs/
│   ├── README.md                # This file
│   ├── requirements.txt
│   └── feature_importance.png   # XGBoost feature importance
│
├── scripts/
│   ├── prepare_features.py      # Data loading, cleaning, categorical encoding
│   ├── embed_text.py            # Sentence-transformer embeddings (chief complaint / admission dx)
│   ├── train.py                 # XGBoost training + internal + external evaluation
│   ├── evaluate.py              # Feature-group ablation study
│   └── nlp_arrest_time.py       # Regex + Stanza NER extraction of arrest times from notes
│
└── data/                        # Empty; place credentialed MIMIC-IV / eICU extracts here
```

Exploration notebooks and the R/DuckDB extraction script are not included in this repository.

Data files (not tracked — require MIMIC-IV and eICU credentialed access):
- `final_data_complete.csv` — MIMIC-IV cohort with all features (79 cols, 19k rows)
- `extra_data_1.csv` — eICU cohort (62 cols, 86k rows)
- `arrest_notes.csv` — Discharge notes for arrest-positive patients
- `vitals_features.csv`, `master_data.csv` — Intermediate MIMIC-IV extracts

---

## Running the Pipeline

```bash
pip install -r docs/requirements.txt

# Step 1: Extract the MIMIC-IV and eICU cohorts (requires credentialed PhysioNet access)
# and save them to data/ as final_data_complete.csv, vitals_features.csv, extra_data_1.csv

# Step 2: Train and evaluate the full model
python scripts/train.py

# Step 3: Ablation study across feature groups
python scripts/evaluate.py

# Step 4: (Optional) Extract arrest times from notes
python scripts/nlp_arrest_time.py
```

The embedding step runs automatically inside `train.py` and `evaluate.py`. It uses Apple MPS by default — edit `DEVICE` in `scripts/embed_text.py` for CUDA or CPU.

---

## Model Details

**Algorithm:** XGBoost (`XGBClassifier`)

**Class imbalance handling:**
- 10:1 downsampling of negatives at training time
- `scale_pos_weight=10` inside XGBoost
- Threshold lowered to 0.3 to prefer recall over precision

**Hyperparameters:** `n_estimators=500`, `learning_rate=0.1`, `max_depth=10`, `subsample=0.8`

**Text embedding:** `sentence-transformers/all-mpnet-base-v2` (768 dims), encoded in batches of 64 on MPS.

---

## Data Access

MIMIC-IV and eICU require credentialed PhysioNet access:
- MIMIC-IV: https://physionet.org/content/mimiciv/
- eICU: https://physionet.org/content/eicu-crd/

---

## Course

LHS 712 — University of Michigan, Winter 2026
