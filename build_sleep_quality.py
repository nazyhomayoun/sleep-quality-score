"""
build_sleep_quality.py
======================
Computes a Sleep Quality Score (SQS) per recording from the epoch-level metadata
produced by build_sleep_dataset.py.

Strategy:
    Some cassette recordings contain sleep scattered across 20+ hours (naps,
    multiple sessions). We identify the LONGEST CONTIGUOUS SLEEP SESSION
    (allowing small wake gaps up to 30 minutes) and compute SQS only on it.

Input:
    processed_4ch/metadata.csv

Output:
    processed_4ch/sleep_quality.csv

Usage:
    python build_sleep_quality.py --input_dir "processed_4ch"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Weight configuration (must sum to 1.0)
# ----------------------------------------------------------------------
WEIGHTS = {
    "se":    0.25,
    "n3":    0.20,
    "rem":   0.15,
    "waso":  0.15,
    "sol":   0.15,
    "frag":  0.10,
}


# ----------------------------------------------------------------------
# Normalization helpers
# ----------------------------------------------------------------------
def _linear_down(x, best, worst):
    if worst == best:
        return 1.0 if x <= best else 0.0
    v = (worst - x) / (worst - best)
    return float(np.clip(v, 0.0, 1.0))


def _triangular(x, peak, low, high):
    if x <= low or x >= high:
        return 0.0
    if x <= peak:
        return float((x - low) / (peak - low))
    return float((high - x) / (high - peak))


def norm_se(se):
    return float(np.clip((se - 0.60) / 0.30, 0.0, 1.0))


def norm_n3(n3_pct):
    return _triangular(n3_pct, peak=18.0, low=0.0, high=40.0)


def norm_rem(rem_pct):
    return _triangular(rem_pct, peak=22.0, low=0.0, high=50.0)


def norm_waso(waso_min):
    return _linear_down(waso_min, best=20.0, worst=120.0)


def norm_sol(sol_min):
    return _linear_down(sol_min, best=15.0, worst=60.0)


def norm_frag(frag_per_hour):
    return _linear_down(frag_per_hour, best=5.0, worst=20.0)


# ----------------------------------------------------------------------
# Sleep session detection
# ----------------------------------------------------------------------
def find_main_sleep_session(stages, epoch_min=0.5,
                            max_gap_min=30.0, min_sleep_min=30.0):
    """
    Find the main sleep session.

    1. Identify all non-W runs (sleep bouts).
    2. Merge bouts separated by < max_gap_min of wake.
    3. Return the longest merged bout (start_idx, end_idx), inclusive.

    Returns None if no usable session is found.
    """
    n = len(stages)
    non_w = (stages != 0).astype(np.int8)

    # Boundaries of non-W runs
    padded = np.concatenate([[0], non_w, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]      # inclusive
    ends = np.where(diff == -1)[0] - 1   # inclusive

    if len(starts) == 0:
        return None

    max_gap_epochs = int(round(max_gap_min / epoch_min))
    min_sleep_epochs = int(round(min_sleep_min / epoch_min))

    # Merge bouts that are separated by <= max_gap_epochs of W
    merged = []
    cur_start, cur_end = starts[0], ends[0]
    for i in range(1, len(starts)):
        if starts[i] - cur_end - 1 <= max_gap_epochs:
            cur_end = ends[i]
        else:
            merged.append((int(cur_start), int(cur_end)))
            cur_start, cur_end = starts[i], ends[i]
    merged.append((int(cur_start), int(cur_end)))

    # Pick the longest merged bout
    lengths = [e - s + 1 for s, e in merged]
    best_idx = int(np.argmax(lengths))
    s, e = merged[best_idx]

    if (e - s + 1) < min_sleep_epochs:
        return None

    return s, e


# ----------------------------------------------------------------------
# Per-record metric computation
# ----------------------------------------------------------------------
def compute_record_metrics(stages, epoch_min=0.5):
    """
    stages: 1D np.ndarray of ints in {0,1,2,3,4}, chronological.
    epoch_min: minutes per epoch (30 sec -> 0.5).
    """
    stages = np.asarray(stages, dtype=int)
    n = len(stages)
    if n == 0:
        return None

    window = find_main_sleep_session(stages, epoch_min=epoch_min)
    if window is None:
        return {
            "n_epochs": n,
            "TIB_min": 0.0,
            "TST_min": 0.0,
            "SE": 0.0,
            "SOL_min": 0.0,
            "WASO_min": 0.0,
            "N1_pct": 0.0, "N2_pct": 0.0, "N3_pct": 0.0, "REM_pct": 0.0,
            "arousal_count": 0,
            "transitions": 0,
            "fragmentation_per_h": 0.0,
            "sleep_quality_score": 0.0,
        }

    onset_idx, offset_idx = window

    # Actual sleep window (main session)
    sleep_window = stages[onset_idx:offset_idx + 1]
    n_in_window = len(sleep_window)

    n_w_in_window = int(np.sum(sleep_window == 0))
    n_sleep = n_in_window - n_w_in_window

    tib_min = n_in_window * epoch_min
    tst_min = n_sleep * epoch_min
    se = tst_min / tib_min if tib_min > 0 else 0.0

    # SOL: from start of recording to sleep onset
    sol_min = onset_idx * epoch_min

    # WASO: wake inside the session
    waso_min = n_w_in_window * epoch_min

    # Stage percentages
    if n_sleep > 0:
        n1_pct = 100.0 * np.sum(sleep_window == 1) / n_sleep
        n2_pct = 100.0 * np.sum(sleep_window == 2) / n_sleep
        n3_pct = 100.0 * np.sum(sleep_window == 3) / n_sleep
        rem_pct = 100.0 * np.sum(sleep_window == 4) / n_sleep
    else:
        n1_pct = n2_pct = n3_pct = rem_pct = 0.0

    # Transitions
    transitions = int(np.sum(sleep_window[1:] != sleep_window[:-1]))

    # Arousals: sleep->W transitions inside the session
    arousal_count = int(np.sum((sleep_window[1:] == 0) & (sleep_window[:-1] != 0)))

    # Fragmentation
    tst_hours = tst_min / 60.0
    frag_per_h = transitions / tst_hours if tst_hours > 0 else 0.0

    # Normalized components
    se_n = norm_se(se)
    n3_n = norm_n3(n3_pct)
    rem_n = norm_rem(rem_pct)
    waso_n = norm_waso(waso_min)
    sol_n = norm_sol(sol_min)
    frag_n = norm_frag(frag_per_h)

    sqs = 100.0 * (
        WEIGHTS["se"] * se_n +
        WEIGHTS["n3"] * n3_n +
        WEIGHTS["rem"] * rem_n +
        WEIGHTS["waso"] * waso_n +
        WEIGHTS["sol"] * sol_n +
        WEIGHTS["frag"] * frag_n
    )

    return {
        "n_epochs": n,
        "sleep_start_epoch": onset_idx,
        "sleep_end_epoch": offset_idx,
        "TIB_min": round(tib_min, 2),
        "TST_min": round(tst_min, 2),
        "SE": round(se, 4),
        "SOL_min": round(sol_min, 2),
        "WASO_min": round(waso_min, 2),
        "N1_pct": round(n1_pct, 2),
        "N2_pct": round(n2_pct, 2),
        "N3_pct": round(n3_pct, 2),
        "REM_pct": round(rem_pct, 2),
        "arousal_count": arousal_count,
        "transitions": transitions,
        "fragmentation_per_h": round(frag_per_h, 2),
        "sleep_quality_score": round(sqs, 2),
    }


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute Sleep Quality Score per recording.")
    parser.add_argument("--input_dir", type=str, default="processed_4ch",
                         help="Folder containing metadata.csv")
    parser.add_argument("--output_csv", type=str, default=None,
                         help="Output CSV path (default: <input_dir>/sleep_quality.csv)")
    parser.add_argument("--max_gap_min", type=float, default=30.0,
                         help="Max wake gap (minutes) to merge sleep bouts into one session")
    parser.add_argument("--min_sleep_min", type=float, default=30.0,
                         help="Minimum duration (minutes) for a valid sleep session")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metadata_path = input_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {input_dir}")

    output_csv = Path(args.output_csv) if args.output_csv else input_dir / "sleep_quality.csv"

    print(f"Reading metadata: {metadata_path}")
    meta = pd.read_csv(metadata_path)
    print(f"Total rows (epochs): {len(meta)}")
    print(f"Unique records: {meta['record_id'].nunique()}")
    print(f"Max wake gap: {args.max_gap_min} min, min session: {args.min_sleep_min} min")

    meta = meta.sort_values(["record_id", "epoch_idx"]).reset_index(drop=True)

    rows = []
    n_total = meta["record_id"].nunique()
    for i, (record_id, group) in enumerate(meta.groupby("record_id", sort=True), start=1):
        stages = group["stage_id"].to_numpy()

        # Override the module-level defaults via a wrapper
        window = find_main_sleep_session(
            stages, epoch_min=0.5,
            max_gap_min=args.max_gap_min,
            min_sleep_min=args.min_sleep_min,
        )

        if window is None:
            metrics = {
                "n_epochs": len(stages), "sleep_start_epoch": -1, "sleep_end_epoch": -1,
                "TIB_min": 0.0, "TST_min": 0.0, "SE": 0.0,
                "SOL_min": 0.0, "WASO_min": 0.0,
                "N1_pct": 0.0, "N2_pct": 0.0, "N3_pct": 0.0, "REM_pct": 0.0,
                "arousal_count": 0, "transitions": 0,
                "fragmentation_per_h": 0.0, "sleep_quality_score": 0.0,
            }
        else:
            # Temporarily patch the defaults by calling a helper that uses these params
            metrics = _compute_with_window(stages, window, epoch_min=0.5)

        first = group.iloc[0]
        row = {
            "record_id": record_id,
            "subject_id": first["subject_id"],
            "night": int(first["night"]),
            "session": first["session"],
        }
        row.update(metrics)
        rows.append(row)

        if i % 20 == 0 or i == n_total:
            print(f"[{i}/{n_total}] processed {record_id}  SQS={metrics['sleep_quality_score']}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\n========== Summary ==========")
    print(f"Records processed: {len(df)}")
    print(f"Mean SQS : {df['sleep_quality_score'].mean():.2f}")
    print(f"Median   : {df['sleep_quality_score'].median():.2f}")
    print(f"Min      : {df['sleep_quality_score'].min():.2f}  (record: {df.loc[df['sleep_quality_score'].idxmin(), 'record_id']})")
    print(f"Max      : {df['sleep_quality_score'].max():.2f}  (record: {df.loc[df['sleep_quality_score'].idxmax(), 'record_id']})")
    print(f"\nSaved to: {output_csv}")
    print("\nSQS distribution (deciles):")
    print(df["sleep_quality_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))


# ----------------------------------------------------------------------
# Compute metrics given an explicit sleep window (used by main)
# ----------------------------------------------------------------------
def _compute_with_window(stages, window, epoch_min=0.5):
    onset_idx, offset_idx = window
    sleep_window = stages[onset_idx:offset_idx + 1]
    n_in_window = len(sleep_window)

    n_w_in_window = int(np.sum(sleep_window == 0))
    n_sleep = n_in_window - n_w_in_window

    tib_min = n_in_window * epoch_min
    tst_min = n_sleep * epoch_min
    se = tst_min / tib_min if tib_min > 0 else 0.0

    sol_min = onset_idx * epoch_min
    waso_min = n_w_in_window * epoch_min

    if n_sleep > 0:
        n1_pct = 100.0 * np.sum(sleep_window == 1) / n_sleep
        n2_pct = 100.0 * np.sum(sleep_window == 2) / n_sleep
        n3_pct = 100.0 * np.sum(sleep_window == 3) / n_sleep
        rem_pct = 100.0 * np.sum(sleep_window == 4) / n_sleep
    else:
        n1_pct = n2_pct = n3_pct = rem_pct = 0.0

    transitions = int(np.sum(sleep_window[1:] != sleep_window[:-1]))
    arousal_count = int(np.sum((sleep_window[1:] == 0) & (sleep_window[:-1] != 0)))

    tst_hours = tst_min / 60.0
    frag_per_h = transitions / tst_hours if tst_hours > 0 else 0.0

    se_n = norm_se(se)
    n3_n = norm_n3(n3_pct)
    rem_n = norm_rem(rem_pct)
    waso_n = norm_waso(waso_min)
    sol_n = norm_sol(sol_min)
    frag_n = norm_frag(frag_per_h)

    sqs = 100.0 * (
        WEIGHTS["se"] * se_n +
        WEIGHTS["n3"] * n3_n +
        WEIGHTS["rem"] * rem_n +
        WEIGHTS["waso"] * waso_n +
        WEIGHTS["sol"] * sol_n +
        WEIGHTS["frag"] * frag_n
    )

    return {
        "n_epochs": len(stages),
        "sleep_start_epoch": onset_idx,
        "sleep_end_epoch": offset_idx,
        "TIB_min": round(tib_min, 2),
        "TST_min": round(tst_min, 2),
        "SE": round(se, 4),
        "SOL_min": round(sol_min, 2),
        "WASO_min": round(waso_min, 2),
        "N1_pct": round(n1_pct, 2),
        "N2_pct": round(n2_pct, 2),
        "N3_pct": round(n3_pct, 2),
        "REM_pct": round(rem_pct, 2),
        "arousal_count": arousal_count,
        "transitions": transitions,
        "fragmentation_per_h": round(frag_per_h, 2),
        "sleep_quality_score": round(sqs, 2),
    }


if __name__ == "__main__":
    main()