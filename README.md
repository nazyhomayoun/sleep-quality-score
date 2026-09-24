# PSG-derived Sleep Architecture Score (SQS)

AI-powered estimation of a PSG-derived sleep-architecture index from physiological signals — built for the
5th AI Factory Iran Hackathon ("Karkhooneh").

Dataset: [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) — PhysioNet

---

## Pipeline architecture

```
sleep-edf-database-expanded-1.0.0/   (raw, from PhysioNet)
            │
            ▼
   build_sleep_dataset.py     ── reads PSG.edf + Hypnogram.edf
            │                    splits into 30-second epochs
            ▼                    keeps subject_id for group-wise splitting
   processed_4ch/metadata.csv
   processed_4ch/signals/*.npz
            │
            ▼
   build_sleep_quality.py     ── finds the main sleep session
            │                    computes the Sleep Quality Score (SQS)
            ▼
   processed_4ch/sleep_quality.csv
            │
            ▼
   extract_features.py        ── extracts spectral / Hjorth features per channel
            │
            ▼
   processed_4ch/features.csv
            │
            ▼
   train_model.py              ── trains LightGBM on every channel combination
            │                    GroupKFold on subject_id (no data leakage)
            ▼
   processed_4ch/model_results.csv
   processed_4ch/best_model_predictions.csv
   processed_4ch/feature_importance.csv
            │
            ▼
   streamlit run app.py        ── interactive demo of the results
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run (in this order)

```bash
# 1) Build the epoch-level dataset (4-channel: 2x EEG + EOG + EMG)
python build_sleep_dataset.py \
    --data_dir "sleep-edf-database-expanded-1.0.0" \
    --output_dir "processed_4ch" \
    --channels "EEG Fpz-Cz" "EEG Pz-Oz" "EOG horizontal" "EMG submental"

# 2) Compute the PSG-derived Sleep Architecture Score for every recording
python build_sleep_quality.py --input_dir "processed_4ch"

# 3) Extract signal features per recording
python extract_features.py --input_dir "processed_4ch"

# 4) Optional: tune and evaluate the EEG Pz-Oz LightGBM experiment with nested
#    GroupKFold (outer-test R² is the valid tuning-aware performance estimate)
python tune_lightgbm.py --input_dir "processed_4ch" --telemetry_weights 3 5

# 5) Train and compare the predefined multi-channel and single-channel models
python train_model.py --input_dir "processed_4ch"

# 6) Create out-of-fold SHAP explanations with the same saved sample weights
python explain_shap.py --input_dir "processed_4ch"

# 7) Run the demo
streamlit run app.py
```

## Current data status

> ⚠️ Update this section before the final presentation — after checking
> `failed_records.csv`, fill in the actual number of successfully processed
> records and unique subjects so judges know the scale the pipeline was
> validated on.

- Successful records: `__ / 197`
- Unique subjects (subject_id): `__`
- Sleep stage distribution: see `processed_4ch/metadata.csv`

## Model design

`train_model.py` evaluates every one-, two-, three-, and four-channel
combination of the two EEG, EOG, and EMG channels. Read the winning channel
count from a regenerated `model_results.csv`; do not infer it from a separate
classification experiment.

- **Reference model (multi-channel):** all 4 channels (2x EEG + EOG + EMG)
- **Single-channel model:** `EEG Pz-Oz` or `EEG Fpz-Cz` only — directly
  comparable to the reference model since both come from the same
  feature-extraction pipeline and the same split
- **Validation:** GroupKFold (5 folds) on `subject_id` — no single person
  ever appears in both train and test
- **Hyperparameter tuning:** nested GroupKFold — parameters are selected in
  inner folds and reported performance comes only from untouched outer-test
  subjects. Do not report the inner-CV tuning score as final performance.
- **Metrics:** MAE, RMSE, R², Spearman, Pearson, CCC, score accuracy within
  ±10 points (a descriptive tolerance metric), Bland-Altman

## Scientific scope and limitations (important for Q&A)

**What the target means.** SQS is a researcher-defined, PSG-derived index: it
combines sleep efficiency, N3%, REM%, WASO, sleep-onset latency, and
fragmentation computed from the scored hypnogram. It is not a clinical
diagnosis, a validated clinical sleep-quality instrument, or a patient-reported
outcome such as PSQI.

**How to interpret model performance.** An R² of 0.50 means the signal
features explain approximately 50% of variation in this derived SQS under the
stated subject-wise validation. It does **not** mean “50% clinical accuracy”
and cannot establish that the model predicts perceived or clinical sleep
quality. Since both the target and EEG features come from the same PSG study,
the experiment should be presented as estimation of a hypnogram-derived sleep
architecture index.

**What is required for a clinical claim.** Validate prospectively on an
independent cohort with an outcome collected independently from PSG staging
(for example, PSQI, insomnia severity, clinician assessment, or a prespecified
clinical endpoint). Refit and evaluate against that outcome with subject-wise
external validation.

The "main sleep session" window (`sleep_start_epoch` / `sleep_end_epoch`)
is currently located using the **full** hypnogram — meaning even the
single-channel model still relies on complete multi-channel labels to know
*when* the person was asleep. In a real product built around a single
sensor, this window would need to be detected directly from that one
channel. This is the next engineering step, not something already solved
in the current MVP.

## File structure

| File | Role |
|---|---|
| `build_sleep_dataset.py` | Read PSG+Hypnogram → labeled epochs |
| `build_sleep_quality.py` | Compute the PSG-derived Sleep Architecture Score |
| `extract_features.py` | Extract spectral/Hjorth features from raw signal |
| `train_model.py` | Train, compare models, report metrics |
| `app.py` | Streamlit demo |
| `target_metadata.json` | Machine-readable target definition generated during training |
| `split_dataset.py` | Standalone group-split example (already integrated into build_sleep_dataset) |

## Team

Built for the 5th Karkhooneh Hackathon — AI Factory Iran
In scientific collaboration with the Brain Innovation Center,
Iran University of Medical Sciences
