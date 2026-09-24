"""
explain_shap.py
================
Model explainability (SHAP) for Sleep Quality Score.

Why is this better than feature_importance.csv (produced by train_model.py)?
----------------------------------------------------------------------------
feature_importance.csv only says "overall, which feature is more important
for the model" (gain importance) -- this is a global number, identical for
all records.

SHAP tells, for **each record individually**: "for this particular night,
which features pushed the score up and which pushed it down, and by how
much". This is exactly what hackathon output #05 asks for: "explain which
patterns in the signal the model found important" -- in an interpretable
way for each user/record, not just a global plot.

How it works (without data leakage):
------------------------------------
Exactly like train_model.py, we use GroupKFold on subject_id. For each fold,
we train the model on the train split and compute SHAP values on the test
split (unseen data) -- meaning SHAP for each record comes from a model that
never saw that subject during training, exactly like out-of-fold predictions.

Input:
    processed_4ch/features.csv
    processed_4ch/sleep_quality.csv
    processed_4ch/model_results.csv   (to find the best feature group)

Output:
    processed_4ch/shap_global_importance.csv   <- global feature ranking
    processed_4ch/shap_per_record.csv          <- top 3 factors per record (positive/negative)
    processed_4ch/shap_values.npz              <- full SHAP matrix (for app.py)
    processed_4ch/shap_summary.png             <- summary plot (beeswarm)
    processed_4ch/shap_bar.png                 <- bar plot of global importance

Usage:
    python explain_shap.py --input_dir "processed_4ch"
    python explain_shap.py --input_dir "processed_4ch" --feature_group eeg2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold

RANDOM_STATE = 42
N_SPLITS = 5


# ----------------------------------------------------------------------
# Same feature grouping as used in train_model.py
# (intentionally redefined here so this script is standalone)
# ----------------------------------------------------------------------
def get_feature_groups(feature_cols):
    eeg1 = [c for c in feature_cols if c.startswith("eeg1_")]
    eeg2 = [c for c in feature_cols if c.startswith("eeg2_")]
    eog = [c for c in feature_cols if c.startswith("eog_")]
    emg = [c for c in feature_cols if c.startswith("emg_")]
    return {
        "all_4ch": feature_cols,
        "eeg_both": eeg1 + eeg2,
        "eeg1": eeg1,
        "eeg2": eeg2,
        "eog": eog,
        "emg": emg,
        "eeg1_eog": eeg1 + eog,
        "eeg2_eog": eeg2 + eog,
        "eeg1_emg": eeg1 + emg,
        "eeg2_emg": eeg2 + emg,
    }


def make_model():
    return lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, verbose=-1,
    )


# ----------------------------------------------------------------------
# Compute SHAP out-of-fold (no data leakage)
# ----------------------------------------------------------------------
def compute_oof_shap(X, y, groups, feature_cols):
    """
    For each fold, train the model and return SHAP values of the test records
    of that fold. Final output: SHAP matrix with the same length as X, where
    each row comes from a model that never saw that subject during training.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_samples, n_features = len(y), len(feature_cols)
    shap_matrix = np.zeros((n_samples, n_features), dtype=np.float64)
    base_values = np.zeros(n_samples, dtype=np.float64)
    oof_preds = np.zeros(n_samples, dtype=np.float64)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr = X.iloc[tr_idx][feature_cols]
        X_te = X.iloc[te_idx][feature_cols]
        y_tr = y.iloc[tr_idx]

        model = make_model()
        model.fit(X_tr, y_tr)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_te)
        shap_matrix[te_idx, :] = sv
        base_values[te_idx] = explainer.expected_value
        oof_preds[te_idx] = model.predict(X_te)

        print(f"  fold {fold}/{N_SPLITS}: explained {len(te_idx)} records")

    return shap_matrix, base_values, oof_preds


# ----------------------------------------------------------------------
# Extract top factors (positive/negative) per record -- for demo display
# ----------------------------------------------------------------------
def top_factors_per_record(shap_matrix, feature_cols, top_n=3):
    rows = []
    for i in range(shap_matrix.shape[0]):
        row_shap = shap_matrix[i]
        order = np.argsort(-np.abs(row_shap))[:top_n]
        entry = {}
        for rank, idx in enumerate(order, start=1):
            entry[f"top{rank}_feature"] = feature_cols[idx]
            entry[f"top{rank}_shap"] = round(float(row_shap[idx]), 3)
            entry[f"top{rank}_direction"] = "increases_SQS" if row_shap[idx] > 0 else "decreases_SQS"
        rows.append(entry)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP for SQS model explainability")
    parser.add_argument("--input_dir", type=str, default="processed_4ch")
    parser.add_argument("--feature_group", type=str, default=None,
                         help='Which feature group to explain (e.g. "eeg2"). '
                              'Default: best model according to model_results.csv')
    parser.add_argument("--top_n_plot", type=int, default=15,
                         help="Number of top features in the plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    feats = pd.read_csv(input_dir / "features.csv")
    sq = pd.read_csv(input_dir / "sleep_quality.csv")
    df = feats.merge(sq[["record_id", "sleep_quality_score"]], on="record_id", how="inner")
    print(f"Merged records: {len(df)}  |  Unique subjects: {df['subject_id'].nunique()}")

    feature_cols_all = [c for c in feats.columns
                         if c not in ("record_id", "subject_id", "night", "session")]
    groups_map = get_feature_groups(feature_cols_all)

    # ---- Choose feature group ----
    if args.feature_group:
        if args.feature_group not in groups_map:
            raise ValueError(f"Group '{args.feature_group}' not found. Options: {list(groups_map)}")
        chosen = args.feature_group
    else:
        results_path = input_dir / "model_results.csv"
        if results_path.exists():
            results = pd.read_csv(results_path).sort_values("MAE")
            chosen = results.iloc[0]["model"]
            print(f"Best model according to model_results.csv: {chosen}")
        else:
            chosen = "all_4ch"
            print(f"⚠️ model_results.csv not found -- using default group: {chosen}")

    feature_cols = groups_map[chosen]
    print(f"Explainability for feature group: {chosen}  ({len(feature_cols)} features)")

    X = df
    y = df["sleep_quality_score"]
    groups = df["subject_id"]

    print("\nComputing SHAP out-of-fold (no data leakage)...")
    shap_matrix, base_values, oof_preds = compute_oof_shap(X, y, groups, feature_cols)

    # ---- Global feature importance ----
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    global_imp = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    global_imp.to_csv(input_dir / "shap_global_importance.csv", index=False)
    print(f"\nTop {min(10, len(global_imp))} features (mean |SHAP|):")
    print(global_imp.head(10).to_string(index=False))

    # ---- Per-record explanations ----
    per_record = top_factors_per_record(shap_matrix, feature_cols, top_n=3)
    per_record.insert(0, "record_id", df["record_id"].values)
    per_record.insert(1, "subject_id", df["subject_id"].values)
    per_record.insert(2, "session", df["session"].values)
    per_record.insert(3, "actual_sqs", df["sleep_quality_score"].values)
    per_record.insert(4, "predicted_sqs_oof", oof_preds)
    per_record.insert(5, "base_value", base_values)
    per_record.to_csv(input_dir / "shap_per_record.csv", index=False)

    # ---- Save full matrix for use in app.py ----
    np.savez_compressed(
        input_dir / "shap_values.npz",
        shap_matrix=shap_matrix,
        base_values=base_values,
        feature_cols=np.array(feature_cols),
        record_ids=df["record_id"].values,
    )

    # ---- Bar plot of global importance ----
    # Note: titles/labels intentionally kept in English because matplotlib
    # without bidi/arabic-reshaper renders Persian text reversed or broken.
    top = global_imp.head(args.top_n_plot).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#3498DB")
    plt.xlabel("Mean |SHAP value| (impact on SQS)")
    plt.title(f"Top features -- group: {chosen}")
    plt.tight_layout()
    plt.savefig(input_dir / "shap_bar.png", dpi=150)
    plt.close()

    # ---- Beeswarm plot (also shows direction of each feature's effect) ----
    try:
        plt.figure()
        shap.summary_plot(
            shap_matrix, X[feature_cols], show=False, max_display=args.top_n_plot
        )
        plt.tight_layout()
        plt.savefig(input_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"⚠️ Beeswarm plot failed (bar plot was still saved): {e}")

    print("\n========== Summary ==========")
    print(f"Feature group explained : {chosen}")
    print(f"Number of records       : {len(df)}")
    print(f"shap_global_importance.csv  saved")
    print(f"shap_per_record.csv         saved (top 3 factors per record)")
    print(f"shap_values.npz             saved (for app.py)")
    print(f"shap_bar.png / shap_summary.png  saved")


if __name__ == "__main__":
    main()