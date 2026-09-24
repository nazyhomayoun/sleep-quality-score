"""
train_model.py
==============
Trains and evaluates regression models to predict Sleep Quality Score (SQS)
from physiological features.

Primary metrics (regression, not classification):
    - MAE, RMSE  (error magnitude)
    - R²         (variance explained)
    - Spearman   (rank correlation)
    - CCC        (concordance correlation coefficient)
    - Clinical Accuracy (±10 points)  -> interpretable for clinicians
    - Bland-Altman bias and limits of agreement

Domain adaptation:
    --telemetry_weight W  -> telemetry samples get weight W in training.
                             Use W > 1 to make the model focus on telemetry
                             (the hackathon test domain).

Validation:
    GroupKFold on subject_id (prevents leakage between nights of the same subject)

Usage:
    python train_model.py --input_dir "processed_4ch" --telemetry_weight 3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import lightgbm as lgb


RANDOM_STATE = 42
N_SPLITS = 5
CLINICAL_THRESHOLD = 10.0


# ----------------------------------------------------------------------
# Clinical / regression metrics
# ----------------------------------------------------------------------
def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mt) * (y_pred - mp))
    denom = vt + vp + (mt - mp) ** 2
    return float(2 * cov / denom) if denom > 0 else 0.0


def clinical_accuracy(y_true, y_pred, threshold=CLINICAL_THRESHOLD):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(100.0 * np.mean(np.abs(y_true - y_pred) <= threshold))


def bland_altman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def full_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    sp = spearmanr(y_true, y_pred).correlation if len(y_true) > 2 else 0.0
    pe = pearsonr(y_true, y_pred)[0] if len(y_true) > 2 else 0.0
    ccc_val = ccc(y_true, y_pred)
    clin_acc = clinical_accuracy(y_true, y_pred, CLINICAL_THRESHOLD)
    bias, loa_lo, loa_up = bland_altman(y_true, y_pred)

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
        "Spearman": round(float(sp) if not np.isnan(sp) else 0.0, 3),
        "Pearson": round(float(pe) if not np.isnan(pe) else 0.0, 3),
        "CCC": round(ccc_val, 3),
        "ClinAcc_10": round(clin_acc, 1),
        "BA_bias": round(bias, 2),
        "BA_LoA_lower": round(loa_lo, 2),
        "BA_LoA_upper": round(loa_up, 2),
    }


# ----------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------
def get_feature_groups(feature_cols):
    eeg1 = [c for c in feature_cols if c.startswith("eeg1_")]
    eeg2 = [c for c in feature_cols if c.startswith("eeg2_")]
    eog  = [c for c in feature_cols if c.startswith("eog_")]
    emg  = [c for c in feature_cols if c.startswith("emg_")]
    return {
        "all_4ch":  feature_cols,
        "eeg_both": eeg1 + eeg2,
        "eeg1":     eeg1,
        "eeg2":     eeg2,
        "eog":      eog,
        "emg":      emg,
        "eeg1_eog": eeg1 + eog,
        "eeg2_eog": eeg2 + eog,
        "eeg1_emg": eeg1 + emg,
        "eeg2_emg": eeg2 + emg,
    }


# ----------------------------------------------------------------------
# Cross-validated evaluation (with sample weights)
# ----------------------------------------------------------------------
def evaluate_model(X, y, groups, feature_cols, model_name, sample_weights):
    gkf = GroupKFold(n_splits=N_SPLITS)
    all_preds = np.zeros(len(y), dtype=float)
    fold_models = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr = X.iloc[tr_idx][feature_cols]
        X_te = X.iloc[te_idx][feature_cols]
        y_tr = y.iloc[tr_idx]
        w_tr = sample_weights[tr_idx]

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        all_preds[te_idx] = model.predict(X_te)
        fold_models.append(model)

    metrics = full_metrics(y.values, all_preds)
    metrics["model"] = model_name
    metrics["n_features"] = len(feature_cols)
    return metrics, all_preds, fold_models


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SQS regression models.")
    parser.add_argument("--input_dir", type=str, default="processed_4ch")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--telemetry_weight", type=float, default=1.0,
                         help="Sample weight for telemetry records during training. "
                              "1.0 = no weighting, 3.0 = telemetry counts 3x more.")
    parser.add_argument("--cassette_weight", type=float, default=1.0,
                         help="Sample weight for cassette records during training.")
    parser.add_argument("--only_telemetry", action="store_true",
                         help="Train and evaluate ONLY on telemetry records.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(input_dir / "features.csv")
    sq = pd.read_csv(input_dir / "sleep_quality.csv")

    df = feats.merge(sq[["record_id", "sleep_quality_score"]],
                     on="record_id", how="inner")
    print(f"Merged rows: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"Sessions: {df['session'].value_counts().to_dict()}")

    # ---- Optional: keep only telemetry ----
    if args.only_telemetry:
        df = df[df["session"] == "telemetry"].reset_index(drop=True)
        print(f"⚠️  only_telemetry=True -> using {len(df)} records")

    feature_cols = [c for c in feats.columns
                    if c not in ("record_id", "subject_id", "night", "session")]

    X = df
    y = df["sleep_quality_score"]
    groups = df["subject_id"]

    # ---- Sample weights ----
    sample_weights = np.where(
        df["session"].values == "telemetry",
        args.telemetry_weight,
        args.cassette_weight,
    )
    print(f"\nSample weights: cassette={args.cassette_weight}, "
          f"telemetry={args.telemetry_weight}")

    groups_map = get_feature_groups(feature_cols)
    print("\nFeature groups:")
    for name, cols in groups_map.items():
        print(f"  {name}: {len(cols)} features")

    # ---- Train every model ----
    results = []
    predictions = {}
    fold_models_dict = {}

    for name, cols in groups_map.items():
        print(f"\n=== {name} ({len(cols)} features) ===")
        metrics, preds, fold_models = evaluate_model(
            X, y, groups, cols, name, sample_weights)
        results.append(metrics)
        predictions[name] = preds
        fold_models_dict[name] = fold_models
        print(f"  MAE={metrics['MAE']:.2f}  R²={metrics['R2']:.3f}  "
              f"Spearman={metrics['Spearman']:.3f}  CCC={metrics['CCC']:.3f}  "
              f"ClinAcc±10={metrics['ClinAcc_10']:.1f}%")

    # ---- Sort by MAE ----
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE", ascending=True).reset_index(drop=True)
    results_df.to_csv(output_dir / "model_results.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_preds = predictions[best_name]
    print(f"\n========== Best model by MAE: {best_name} ==========")

    # ---- Comparison table ----
    print("\n========== Model comparison (sorted by MAE) ==========")
    cols_to_show = ["model", "n_features", "MAE", "RMSE", "R2",
                    "Spearman", "CCC", "ClinAcc_10"]
    print(results_df[cols_to_show].to_string(index=False))

    # ---- Save predictions of best model ----
    pred_df = df[["record_id", "subject_id", "session", "sleep_quality_score"]].copy()
    pred_df["predicted_sqs"] = best_preds
    pred_df["residual"] = pred_df["sleep_quality_score"] - pred_df["predicted_sqs"]
    pred_df["abs_error"] = pred_df["residual"].abs()
    pred_df["within_10"] = (pred_df["abs_error"] <= CLINICAL_THRESHOLD).astype(int)
    pred_df.to_csv(output_dir / "best_model_predictions.csv", index=False)

    # ---- Per-session breakdown ----
    print(f"\n========== Per-session breakdown ({best_name}) ==========")
    session_rows = []
    for sess in ["cassette", "telemetry"]:
        mask = pred_df["session"].values == sess
        if mask.sum() == 0:
            continue
        m = full_metrics(
            pred_df.loc[mask, "sleep_quality_score"].values,
            pred_df.loc[mask, "predicted_sqs"].values,
        )
        m["session"] = sess
        m["n"] = int(mask.sum())
        session_rows.append(m)
        print(f"  {sess:10s}  n={m['n']:3d}  MAE={m['MAE']:.2f}  R²={m['R2']:.3f}  "
              f"Spearman={m['Spearman']:.3f}  ClinAcc±10={m['ClinAcc_10']:.1f}%")

    if session_rows:
        pd.DataFrame(session_rows).to_csv(
            output_dir / "session_breakdown.csv", index=False)

    # ---- Feature importance ----
    best_cols = groups_map[best_name]
    if len(best_cols) > 0:
        final_model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=15,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, verbose=-1,
        )
        final_model.fit(X[best_cols], y, sample_weight=sample_weights)
        imps = final_model.booster_.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({
            "feature": best_cols,
            "gain_importance": imps,
        }).sort_values("gain_importance", ascending=False).reset_index(drop=True)
        imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

        print(f"\nTop 15 features by gain ({best_name}):")
        print(imp_df.head(15).to_string(index=False))

    # ---- Final summary ----
    best_metrics = results_df.iloc[0]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"telemetry_weight      : {args.telemetry_weight}")
    print(f"cassette_weight       : {args.cassette_weight}")
    print(f"only_telemetry        : {args.only_telemetry}")
    print(f"Winning model         : {best_name} ({int(best_metrics['n_features'])} features)")
    print(f"MAE                   : {best_metrics['MAE']:.2f} SQS points")
    print(f"R²                    : {best_metrics['R2']:.3f}")
    print(f"Spearman              : {best_metrics['Spearman']:.3f}")
    print(f"CCC                   : {best_metrics['CCC']:.3f}")
    print(f"Clinical Accuracy     : {best_metrics['ClinAcc_10']:.1f}% within ±10")
    print("=" * 60)


if __name__ == "__main__":
    main()