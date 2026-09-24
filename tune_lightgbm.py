"""
tune_lightgbm.py
================
Grid search over LightGBM hyperparameters for best R².
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from itertools import product


INPUT_DIR = Path("processed_4ch")
N_SPLITS = 5


def cv_r2(X, y, groups, cols, params, sample_weights):
    gkf = GroupKFold(n_splits=N_SPLITS)
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        m.fit(X.iloc[tr][cols], y.iloc[tr], sample_weight=sample_weights[tr])
        preds[te] = m.predict(X.iloc[te][cols])
    return r2_score(y, preds)


def main():
    feats = pd.read_csv(INPUT_DIR / "features.csv")
    sq = pd.read_csv(INPUT_DIR / "sleep_quality.csv")
    df = feats.merge(sq[["record_id", "sleep_quality_score"]], on="record_id")

    cols = [c for c in feats.columns if c.startswith("eeg2_")]
    X, y, g = df, df["sleep_quality_score"], df["subject_id"]
    sw = np.where(df["session"].values == "telemetry", 5.0, 1.0)

    # Grid
    grid = {
        "n_estimators":   [300, 500, 800],
        "learning_rate":  [0.02, 0.03, 0.05],
        "num_leaves":     [5, 7, 15],
        "min_child_samples": [10, 15, 20],
    }

    combos = list(product(*grid.values()))
    keys = list(grid.keys())
    print(f"Testing {len(combos)} combinations...\n")

    best_r2, best_params = -np.inf, None
    results = []

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        r2 = cv_r2(X, y, g, cols, params, sw)
        results.append({**params, "R2": r2})
        if r2 > best_r2:
            best_r2, best_params = r2, params
            print(f"[{i:3d}/{len(combos)}] R²={r2:.4f} ← NEW BEST  {params}")
        else:
            print(f"[{i:3d}/{len(combos)}] R²={r2:.4f}")

    print(f"\n{'='*60}")
    print(f"BEST R² = {best_r2:.4f}")
    print(f"BEST PARAMS = {best_params}")
    print(f"{'='*60}")

    # Save
    res_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    res_df.to_csv(INPUT_DIR / "tuning_results.csv", index=False)
    print(f"\nTop 5:")
    print(res_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()