"""Nested GroupKFold tuning with unbiased outer-CV evaluation for SQS."""

import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT_DIR = Path("processed_4ch")
RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4


def make_model(params):
    return lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1)


def grouped_cv_r2(X, y, groups, cols, params, sample_weights, n_splits):
    """Return out-of-fold R² using only the supplied partition."""
    cv = GroupKFold(n_splits=n_splits)
    predictions = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y, groups):
        model = make_model(params)
        model.fit(
            X.iloc[train_idx][cols], y.iloc[train_idx],
            sample_weight=sample_weights[train_idx],
        )
        predictions[test_idx] = model.predict(X.iloc[test_idx][cols])
    return r2_score(y, predictions)


def validate_split_count(groups, n_splits, split_name):
    n_groups = pd.Series(groups).nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"{split_name} GroupKFold needs at least {n_splits} subjects; found {n_groups}."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Nested GroupKFold tuning and unbiased outer-CV evaluation."
    )
    parser.add_argument("--input_dir", type=str, default=str(INPUT_DIR))
    parser.add_argument("--outer_splits", type=int, default=OUTER_SPLITS)
    parser.add_argument("--inner_splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--telemetry_weights", type=float, nargs="+", default=[3.0, 5.0],
                        help="Candidate telemetry weights selected inside inner CV.")
    args = parser.parse_args()

    if args.outer_splits < 2 or args.inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must both be at least 2.")

    input_dir = Path(args.input_dir)
    feats = pd.read_csv(input_dir / "features.csv")
    sq = pd.read_csv(input_dir / "sleep_quality.csv")
    df = feats.merge(sq[["record_id", "sleep_quality_score"]], on="record_id")

    cols = [column for column in feats.columns if column.startswith("eeg2_")]
    X = df.reset_index(drop=True)
    y = X["sleep_quality_score"].reset_index(drop=True)
    groups = X["subject_id"].reset_index(drop=True)
    validate_split_count(groups, args.outer_splits, "Outer")

    grid = {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.02, 0.03, 0.05],
        "num_leaves": [5, 7, 15],
        "min_child_samples": [10, 15, 20],
    }
    keys = list(grid)
    parameter_sets = [dict(zip(keys, values)) for values in product(*grid.values())]

    print(
        f"Nested GroupKFold: {args.outer_splits} outer folds, "
        f"{args.inner_splits} inner folds, {len(parameter_sets)} parameter sets, "
        f"and telemetry weights {args.telemetry_weights}."
    )
    print("Outer test subjects are excluded from hyperparameter selection.\n")

    outer_cv = GroupKFold(n_splits=args.outer_splits)
    outer_predictions = np.zeros(len(y))
    inner_rows, outer_rows = [], []

    for fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        X_train = X.iloc[outer_train_idx].reset_index(drop=True)
        y_train = y.iloc[outer_train_idx].reset_index(drop=True)
        groups_train = groups.iloc[outer_train_idx].reset_index(drop=True)
        validate_split_count(groups_train, args.inner_splits, f"Inner fold {fold}")

        best_inner_r2, best_params, best_weight = -np.inf, None, None
        for telemetry_weight in args.telemetry_weights:
            weights_train = np.where(
                X_train["session"].values == "telemetry", telemetry_weight, 1.0
            )
            for params in parameter_sets:
                inner_r2 = grouped_cv_r2(
                    X_train, y_train, groups_train, cols, params, weights_train, args.inner_splits
                )
                inner_rows.append({
                    "outer_fold": fold, "telemetry_weight": telemetry_weight,
                    **params, "inner_R2": inner_r2,
                })
                if inner_r2 > best_inner_r2:
                    best_inner_r2, best_params, best_weight = inner_r2, params, telemetry_weight

        final_model = make_model(best_params)
        outer_train_weights = np.where(
            X.iloc[outer_train_idx]["session"].values == "telemetry", best_weight, 1.0
        )
        final_model.fit(
            X.iloc[outer_train_idx][cols], y.iloc[outer_train_idx],
            sample_weight=outer_train_weights,
        )
        fold_predictions = final_model.predict(X.iloc[outer_test_idx][cols])
        outer_predictions[outer_test_idx] = fold_predictions
        outer_r2 = r2_score(y.iloc[outer_test_idx], fold_predictions)
        outer_mae = mean_absolute_error(y.iloc[outer_test_idx], fold_predictions)
        outer_rmse = float(np.sqrt(mean_squared_error(y.iloc[outer_test_idx], fold_predictions)))
        outer_rows.append({
            "outer_fold": fold,
            "n_train": len(outer_train_idx), "n_test": len(outer_test_idx),
            "n_train_subjects": groups.iloc[outer_train_idx].nunique(),
            "n_test_subjects": groups.iloc[outer_test_idx].nunique(),
            "best_inner_R2": best_inner_r2, "outer_test_R2": outer_r2,
            "outer_test_MAE": outer_mae, "outer_test_RMSE": outer_rmse,
            "telemetry_weight": best_weight,
            **best_params,
        })
        print(
            f"Outer fold {fold}/{args.outer_splits}: inner-best R²={best_inner_r2:.4f}; "
            f"held-out outer-test R²={outer_r2:.4f}; weight={best_weight}; params={best_params}"
        )

    nested_r2 = r2_score(y, outer_predictions)
    nested_mae = mean_absolute_error(y, outer_predictions)
    nested_rmse = float(np.sqrt(mean_squared_error(y, outer_predictions)))
    nested_within_10 = float(np.mean(np.abs(y.values - outer_predictions) <= 10.0) * 100.0)
    outer_df = pd.DataFrame(outer_rows)
    inner_df = pd.DataFrame(inner_rows)
    prediction_df = X[["record_id", "subject_id", "session", "sleep_quality_score"]].copy()
    prediction_df["nested_cv_prediction"] = outer_predictions
    prediction_df["residual"] = prediction_df["sleep_quality_score"] - outer_predictions
    metrics = {
        "n_records": len(y),
        "nested_outer_cv_R2": nested_r2,
        "nested_outer_cv_MAE": nested_mae,
        "nested_outer_cv_RMSE": nested_rmse,
        "within_10_points_pct": nested_within_10,
    }

    # These are inner-CV scores, not final-performance estimates.
    inner_df.to_csv(input_dir / "tuning_results.csv", index=False)
    outer_df.to_csv(input_dir / "nested_cv_outer_folds.csv", index=False)
    prediction_df.to_csv(input_dir / "nested_cv_predictions.csv", index=False)
    with open(input_dir / "nested_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # This selection is for the final all-data refit only. Its CV score is not
    # a performance claim; the nested outer-CV score above is that estimate.
    deployment_ranking = (
        inner_df.groupby(["telemetry_weight", *keys], as_index=False)["inner_R2"].mean()
        .sort_values("inner_R2", ascending=False)
        .reset_index(drop=True)
    )
    deployment_weight = float(deployment_ranking.iloc[0]["telemetry_weight"])
    deployment_params = deployment_ranking.iloc[0][keys].to_dict()
    deployment_params = {
        key: int(value) if key in {"n_estimators", "num_leaves", "min_child_samples"}
        else float(value)
        for key, value in deployment_params.items()
    }
    with open(input_dir / "lightgbm_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "params": deployment_params,
            "telemetry_weight": deployment_weight,
            "selection_note": "Deployment-only parameters; report NESTED OUTER-CV R², not inner_R2.",
        }, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f"NESTED OUTER-CV R² = {nested_r2:.4f}")
    print(f"NESTED OUTER-CV MAE = {nested_mae:.3f}")
    print(f"NESTED OUTER-CV RMSE = {nested_rmse:.3f}")
    print(f"Within ±10 points = {nested_within_10:.1f}%")
    print("This is the performance estimate after hyperparameter tuning.")
    print(f"Mean outer-fold R² = {outer_df['outer_test_R2'].mean():.4f} "
          f"± {outer_df['outer_test_R2'].std(ddof=1):.4f}")
    print(f"Saved inner tuning scores: {input_dir / 'tuning_results.csv'}")
    print(f"Saved outer-fold results: {input_dir / 'nested_cv_outer_folds.csv'}")
    print(f"Saved outer-fold predictions: {input_dir / 'nested_cv_predictions.csv'}")
    print(f"Saved nested metrics: {input_dir / 'nested_cv_metrics.json'}")
    print(f"Saved deployment parameters: {input_dir / 'lightgbm_params.json'}")


if __name__ == "__main__":
    main()
