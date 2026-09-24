"""
improve_r2_final.py
===================
Try 5 strategies to improve R², in order of expected impact:
  1. Baseline (current 21 features)
  2. Feature engineering (+ new ratio features)
  3. Feature selection (SHAP top-K)
  4. Ensemble (5 seeds)
  5. Combined (feature eng + selection + ensemble)

Uses the SAME GroupKFold on subject_id and the SAME hyperparameters
as train_model.py (loaded from lightgbm_params.json).
"""

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr


INPUT_DIR = Path("processed_4ch")
N_SPLITS = 5
TELEMETRY_WEIGHT = 5.0
N_SEEDS = 5


def load_params():
    with open(INPUT_DIR / "lightgbm_params.json") as f:
        data = json.load(f)
    # Handle nested structure: {"params": {...}, "telemetry_weight": ...}
    if "params" in data:
        return data["params"]
    return data


def make_model(params, seed=42):
    p = dict(params)
    p["random_state"] = seed
    return lgb.LGBMRegressor(**p, verbose=-1)


def cv_r2_single_seed(X, y, groups, cols, params, sw):
    gkf = GroupKFold(n_splits=N_SPLITS)
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = make_model(params)
        m.fit(X.iloc[tr][cols], y.iloc[tr], sample_weight=sw[tr])
        preds[te] = m.predict(X.iloc[te][cols])
    return r2_score(y, preds), mean_absolute_error(y, preds), spearmanr(y, preds).correlation


def cv_r2_ensemble(X, y, groups, cols, params, sw, n_seeds=N_SEEDS):
    gkf = GroupKFold(n_splits=N_SPLITS)
    all_seed_preds = np.zeros((n_seeds, len(y)))
    for s in range(n_seeds):
        preds = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            m = make_model(params, seed=s)
            m.fit(X.iloc[tr][cols], y.iloc[tr], sample_weight=sw[tr])
            preds[te] = m.predict(X.iloc[te][cols])
        all_seed_preds[s] = preds
    final = all_seed_preds.mean(axis=0)
    return r2_score(y, final), mean_absolute_error(y, final), spearmanr(y, final).correlation


def main():
    # Load data
    feats = pd.read_csv(INPUT_DIR / "features.csv")
    sq = pd.read_csv(INPUT_DIR / "sleep_quality.csv")
    df = feats.merge(sq[["record_id", "sleep_quality_score"]], on="record_id")

    # ---- Feature Engineering ----
    # Add physiological ratios that are likely informative
    df["eeg2_delta_sigma_ratio"] = df["eeg2_delta_abs"] / (df["eeg2_sigma_abs"] + 1e-6)
    df["eeg2_delta_alpha_ratio"] = df["eeg2_delta_abs"] / (df["eeg2_alpha_abs"] + 1e-6)
    df["eeg2_sigma_delta_ratio"] = df["eeg2_sigma_abs"] / (df["eeg2_delta_abs"] + 1e-6)
    df["eeg2_theta_sigma_ratio"] = df["eeg2_theta_abs"] / (df["eeg2_sigma_abs"] + 1e-6)
    df["eeg2_slow_sigma_ratio"] = (df["eeg2_delta_abs"] + df["eeg2_theta_abs"]) / (df["eeg2_sigma_abs"] + 1e-6)
    df["eeg2_deep_shallow"] = df["eeg2_delta_abs"] / (df["eeg2_alpha_abs"] + df["eeg2_beta_abs"] + 1e-6)

    print(f"Feature engineering added 6 new features")

    y = df["sleep_quality_score"]
    groups = df["subject_id"]
    sw = np.where(df["session"].values == "telemetry", TELEMETRY_WEIGHT, 1.0)

    params = load_params()
    # ensure random_state is set for reproducibility
    params = {**params, "random_state": 42}
    print(f"Using tuned params: {params}\n")

    # Feature sets
    base_cols = [c for c in feats.columns if c.startswith("eeg2_")]  # 21
    eng_cols = base_cols + [c for c in df.columns if c.startswith("eeg2_") and c not in base_cols]  # 27

    # Top-K selection from SHAP (or fallback to gain)
    shap_path = INPUT_DIR / "shap_global_importance.csv"
    if shap_path.exists():
        imp = pd.read_csv(shap_path)
        imp_cols = [c for c in imp["feature"].tolist() if c in eng_cols]
    else:
        m = make_model(params)
        m.fit(df[eng_cols], y, sample_weight=sw)
        gains = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=eng_cols)
        imp_cols = gains.sort_values(ascending=False).index.tolist()

    results = []

    # ---- 1. Baseline ----
    print("=" * 60)
    print("1. Baseline (21 features, single seed)")
    print("=" * 60)
    r2, mae, sp = cv_r2_single_seed(df, y, groups, base_cols, params, sw)
    print(f"   R²={r2:.4f}  MAE={mae:.2f}  Spearman={sp:.4f}")
    results.append(("baseline 21 feat", r2, mae, sp))

    # ---- 2. Feature Engineering ----
    print("\n" + "=" * 60)
    print("2. Feature Engineering (27 features, single seed)")
    print("=" * 60)
    r2, mae, sp = cv_r2_single_seed(df, y, groups, eng_cols, params, sw)
    print(f"   R²={r2:.4f}  MAE={mae:.2f}  Spearman={sp:.4f}")
    results.append(("feat eng 27 feat", r2, mae, sp))

    # ---- 3. Feature Selection (top K from importance) ----
    print("\n" + "=" * 60)
    print("3. Feature Selection (top-K from importance)")
    print("=" * 60)
    for k in [8, 10, 12, 15]:
        cols_k = imp_cols[:k]
        r2, mae, sp = cv_r2_single_seed(df, y, groups, cols_k, params, sw)
        print(f"   top {k:2d}: R²={r2:.4f}  MAE={mae:.2f}  Spearman={sp:.4f}")
        results.append((f"top {k} feat", r2, mae, sp))

    # ---- 4. Ensemble (5 seeds) on baseline ----
    print("\n" + "=" * 60)
    print("4. Ensemble (5 seeds, 21 features)")
    print("=" * 60)
    r2, mae, sp = cv_r2_ensemble(df, y, groups, base_cols, params, sw)
    print(f"   R²={r2:.4f}  MAE={mae:.2f}  Spearman={sp:.4f}")
    results.append(("ensemble 21 feat", r2, mae, sp))

    # ---- 5. Combined: feat eng + top-K + ensemble ----
    print("\n" + "=" * 60)
    print("5. Combined: feat eng + top-K + ensemble")
    print("=" * 60)
    for k in [12, 15, 20]:
        cols_k = imp_cols[:k]
        r2, mae, sp = cv_r2_ensemble(df, y, groups, cols_k, params, sw)
        print(f"   top {k:2d} + ensemble: R²={r2:.4f}  MAE={mae:.2f}  Spearman={sp:.4f}")
        results.append((f"top {k} + ensemble", r2, mae, sp))

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY — sorted by R²")
    print("=" * 70)
    res_df = pd.DataFrame(results, columns=["strategy", "R2", "MAE", "Spearman"])
    res_df = res_df.sort_values("R2", ascending=False).reset_index(drop=True)
    print(res_df.to_string(index=False))

    best = res_df.iloc[0]
    print(f"\n🏆 Best: {best['strategy']}")
    print(f"   R²       = {best['R2']:.4f}")
    print(f"   MAE      = {best['MAE']:.2f}")
    print(f"   Spearman = {best['Spearman']:.4f}")

    res_df.to_csv(INPUT_DIR / "improve_r2_results.csv", index=False)
    print(f"\nSaved: {INPUT_DIR / 'improve_r2_results.csv'}")


if __name__ == "__main__":
    main()