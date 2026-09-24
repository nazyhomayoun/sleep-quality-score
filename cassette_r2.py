"""
cassette_r2.py
==============
R² only. Cassette only. Feature selection included.
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


def make_model():
    return lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, verbose=-1,
    )


def get_r2(X, y, groups, feature_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    preds = np.zeros(len(y))
    for tr_idx, te_idx in gkf.split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr_idx][feature_cols], y.iloc[tr_idx])
        preds[te_idx] = m.predict(X.iloc[te_idx][feature_cols])
    return r2_score(y, preds)


def nested_feature_selection_r2(X, y, groups, candidate_cols, top_ks):
    """Evaluate top-K models with feature selection fit inside each CV fold."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    predictions = {k: np.zeros(len(y)) for k in top_ks}
    fold_importances = []

    for tr_idx, te_idx in gkf.split(X, y, groups):
        # Rank features with training subjects only; held-out subjects cannot
        # influence either importance scores or the selected feature set.
        selector = make_model()
        selector.fit(X.iloc[tr_idx][candidate_cols], y.iloc[tr_idx])
        importances = pd.Series(
            selector.booster_.feature_importance(importance_type="gain"),
            index=candidate_cols,
        )
        fold_importances.append(importances)
        ranked_cols = importances.sort_values(ascending=False).index

        for k in top_ks:
            selected_cols = ranked_cols[:k].tolist()
            model = make_model()
            model.fit(X.iloc[tr_idx][selected_cols], y.iloc[tr_idx])
            predictions[k][te_idx] = model.predict(X.iloc[te_idx][selected_cols])

    mean_importances = pd.DataFrame(fold_importances).mean().sort_values(ascending=False)
    scores = {k: r2_score(y, preds) for k, preds in predictions.items()}
    return scores, mean_importances


def main():
    feats = pd.read_csv(INPUT_DIR / "features.csv")
    sq = pd.read_csv(INPUT_DIR / "sleep_quality.csv")
    df = feats.merge(sq[["record_id", "sleep_quality_score"]], on="record_id")

    # Cassette only
    df = df[df["session"] == "cassette"].reset_index(drop=True)
    print(f"Cassette records: {len(df)}  |  Subjects: {df['subject_id'].nunique()}")
    print(f"SQS: {df['sleep_quality_score'].min():.1f} - {df['sleep_quality_score'].max():.1f}  "
          f"(mean {df['sleep_quality_score'].mean():.2f})\n")

    all_cols = [c for c in feats.columns
                if c not in ("record_id", "subject_id", "night", "session")]
    eeg2_cols = [c for c in all_cols if c.startswith("eeg2_")]

    X = df
    y = df["sleep_quality_score"]
    groups = df["subject_id"]

    # Baseline
    r2_all = get_r2(X, y, groups, eeg2_cols)
    print(f"All 21 eeg2 features : R² = {r2_all:.4f}")

    # Select features separately within each GroupKFold training split.
    top_ks = [3, 5, 8, 10, 12, 15]
    top_k_r2, imps = nested_feature_selection_r2(X, y, groups, eeg2_cols, top_ks)

    print("\nTop features by mean training-fold gain:")
    for i, (feat, imp) in enumerate(imps.head(15).items(), 1):
        print(f"  {i:2d}. {feat:35s} {imp:10.0f}")

    print("\nNested GroupKFold R² with top-K features:")
    for k in top_ks:
        print(f"  Top {k:2d}: R² = {top_k_r2[k]:.4f}")


if __name__ == "__main__":
    main()
