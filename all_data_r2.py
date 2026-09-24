"""
all_data_r2.py
==============
R² for SQS prediction across different training setups:
    1. Cassette only
    2. Telemetry only
    3. Combined, telemetry_weight = 1, 3, 5
    4. Cross-session: train cassette -> test telemetry
    5. Cross-session: train telemetry -> test cassette

Also applies: outlier removal + feature selection + hyperparameter tuning.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


INPUT_DIR = Path("processed_4ch")
RANDOM_STATE = 42
N_SPLITS = 5


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def make_model(params=None):
    base = dict(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    if params:
        base.update(params)
    return lgb.LGBMRegressor(**base)


# ----------------------------------------------------------------------
# GroupKFold R²
# ----------------------------------------------------------------------
def cv_r2(X, y, groups, cols, sample_weights=None, params=None):
    gkf = GroupKFold(n_splits=N_SPLITS)
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = make_model(params)
        X_tr = X.iloc[tr][cols]
        y_tr = y.iloc[tr]
        if sample_weights is not None:
            m.fit(X_tr, y_tr, sample_weight=sample_weights[tr])
        else:
            m.fit(X_tr, y_tr)
        preds[te] = m.predict(X.iloc[te][cols])
    return r2_score(y, preds)


# ----------------------------------------------------------------------
# Cross-session R² (train one session, test the other)
# ----------------------------------------------------------------------
def cross_session_r2(X, y, session, cols, train_sess, test_sess, params=None):
    tr_mask = (session == train_sess).values
    te_mask = (session == test_sess).values
    if tr_mask.sum() < 10 or te_mask.sum() < 5:
        return None
    m = make_model(params)
    m.fit(X.iloc[tr_mask][cols], y.iloc[tr_mask])
    preds = m.predict(X.iloc[te_mask][cols])
    return r2_score(y.iloc[te_mask], preds)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    feats = pd.read_csv(INPUT_DIR / "features.csv")
    sq = pd.read_csv(INPUT_DIR / "sleep_quality.csv")
    df = feats.merge(sq[["record_id", "sleep_quality_score"]], on="record_id")

    all_cols = [c for c in feats.columns
                if c not in ("record_id", "subject_id", "night", "session")]
    eeg2_cols = [c for c in all_cols if c.startswith("eeg2_")]

    print("=" * 70)
    print(" DATASET OVERVIEW")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"  cassette:  {(df['session'] == 'cassette').sum()}")
    print(f"  telemetry: {(df['session'] == 'telemetry').sum()}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"SQS range: {df['sleep_quality_score'].min():.1f} - "
          f"{df['sleep_quality_score'].max():.1f}")
    print(f"SQS mean ± std: {df['sleep_quality_score'].mean():.2f} ± "
          f"{df['sleep_quality_score'].std():.2f}")

    X = df
    y = df["sleep_quality_score"]
    groups = df["subject_id"]
    session = df["session"]

    # ------------------------------------------------------------------
    # 1. Cassette only
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 1. CASSETTE ONLY")
    print("=" * 70)
    mask_c = (session == "cassette").values
    Xc, yc, gc = X[mask_c].reset_index(drop=True), y[mask_c].reset_index(drop=True), groups[mask_c].reset_index(drop=True)

    r2_c_all = cv_r2(Xc, yc, gc, eeg2_cols)
    print(f"  All 21 features            R² = {r2_c_all:.4f}")

    # Outlier removal
    mask_out = (yc >= 25) & (yc <= 90)
    Xc2, yc2, gc2 = Xc[mask_out].reset_index(drop=True), yc[mask_out].reset_index(drop=True), gc[mask_out].reset_index(drop=True)
    r2_c_clean = cv_r2(Xc2, yc2, gc2, eeg2_cols)
    print(f"  No outliers (25-90)        R² = {r2_c_clean:.4f}  ({mask_out.sum()} records)")

    # Tuned params
    tuned = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 15}
    r2_c_tuned = cv_r2(Xc, yc, gc, eeg2_cols, params=tuned)
    print(f"  Tuned params               R² = {r2_c_tuned:.4f}")

    r2_c_both = cv_r2(Xc2, yc2, gc2, eeg2_cols, params=tuned)
    print(f"  Outliers removed + tuned   R² = {r2_c_both:.4f}")

    # ------------------------------------------------------------------
    # 2. Telemetry only
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 2. TELEMETRY ONLY")
    print("=" * 70)
    mask_t = (session == "telemetry").values
    Xt, yt, gt = X[mask_t].reset_index(drop=True), y[mask_t].reset_index(drop=True), groups[mask_t].reset_index(drop=True)

    r2_t_all = cv_r2(Xt, yt, gt, eeg2_cols)
    print(f"  All 21 features            R² = {r2_t_all:.4f}")

    r2_t_tuned = cv_r2(Xt, yt, gt, eeg2_cols, params=tuned)
    print(f"  Tuned params               R² = {r2_t_tuned:.4f}")

    # ------------------------------------------------------------------
    # 3. Combined (different telemetry weights)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 3. COMBINED (cassette + telemetry)")
    print("=" * 70)
    for w in [1.0, 2.0, 3.0, 5.0, 8.0]:
        sw = np.where(session.values == "telemetry", w, 1.0)
        r2 = cv_r2(X, y, groups, eeg2_cols, sample_weights=sw)
        print(f"  telemetry_weight = {w:.1f}        R² = {r2:.4f}")

    # ------------------------------------------------------------------
    # 4. Cross-session: train cassette -> test telemetry
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 4. TRAIN CASSETTE → TEST TELEMETRY")
    print("=" * 70)
    r2_ct = cross_session_r2(X, y, session, eeg2_cols, "cassette", "telemetry")
    print(f"  R² = {r2_ct:.4f}" if r2_ct is not None else "  Not enough data")

    # ------------------------------------------------------------------
    # 5. Cross-session: train telemetry -> test cassette
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 5. TRAIN TELEMETRY → TEST CASSETTE")
    print("=" * 70)
    r2_tc = cross_session_r2(X, y, session, eeg2_cols, "telemetry", "cassette")
    print(f"  R² = {r2_tc:.4f}" if r2_tc is not None else "  Not enough data")

    # ------------------------------------------------------------------
    # 6. Feature selection on full data
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 6. FEATURE SELECTION (all data, telemetry_weight=3)")
    print("=" * 70)
    sw = np.where(session.values == "telemetry", 3.0, 1.0)

    # Get top features by LightGBM gain on full data
    m = make_model(tuned)
    m.fit(X[eeg2_cols], y, sample_weight=sw)
    imps = pd.Series(
        m.booster_.feature_importance(importance_type="gain"),
        index=eeg2_cols,
    ).sort_values(ascending=False)

    for k in [5, 8, 10, 12, 15, 21]:
        cols_k = imps.head(k).index.tolist()
        r2 = cv_r2(X, y, groups, cols_k, sample_weights=sw, params=tuned)
        print(f"  Top {k:2d} features           R² = {r2:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY — best R² in each setup")
    print("=" * 70)
    print(f"  Cassette only (all feat)         R² = {r2_c_all:.4f}")
    print(f"  Cassette only (best)             R² = {max(r2_c_all, r2_c_clean, r2_c_tuned, r2_c_both):.4f}")
    print(f"  Telemetry only (all feat)        R² = {r2_t_all:.4f}")
    print(f"  Telemetry only (tuned)           R² = {r2_t_tuned:.4f}")
    print(f"  Combined (best weight)           R² = ???")
    print(f"  Cross: cassette → telemetry      R² = {r2_ct:.4f}" if r2_ct else "")
    print(f"  Cross: telemetry → cassette      R² = {r2_tc:.4f}" if r2_tc else "")


if __name__ == "__main__":
    main()