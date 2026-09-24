"""
aasm_compliance.py
==================
Checks how many AASM healthy-adult criteria each recording meets.

AASM healthy adult ranges:
    - SE       > 85%
    - SOL      < 30 min
    - WASO     < 20 min
    - N3%      13-23%
    - REM%     20-25%

Output:
    - Per-record table (aasm_pass, aasm_total)
    - Summary distribution
    - CSV file

Usage:
    python aasm_compliance.py --input_dir "processed_4ch"
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def check_aasm(row):
    """Return number of AASM criteria met (out of 5) and details."""
    checks = {
        "SE>85":     row["SE"] * 100 > 85,
        "SOL<30":    row["SOL_min"] < 30,
        "WASO<20":   row["WASO_min"] < 20,
        "N3_13_23":  13 <= row["N3_pct"] <= 23,
        "REM_20_25": 20 <= row["REM_pct"] <= 25,
    }
    n_pass = sum(checks.values())
    return n_pass, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="processed_4ch")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    df = pd.read_csv(input_dir / "sleep_quality.csv")

    print(f"Total records: {len(df)}\n")

    # Compute AASM compliance for each record
    aasm_pass = []
    detail_rows = []
    for _, row in df.iterrows():
        n_pass, checks = check_aasm(row)
        aasm_pass.append(n_pass)
        detail = {"record_id": row["record_id"]}
        detail.update(checks)
        detail_rows.append(detail)

    df["aasm_pass"] = aasm_pass
    details_df = pd.DataFrame(detail_rows)

    # ---- Distribution table ----
    print("=" * 60)
    print(" AASM COMPLIANCE DISTRIBUTION")
    print("=" * 60)
    counts = df["aasm_pass"].value_counts().sort_index(ascending=False)
    total = len(df)

    print(f"\n{'Criteria met':<15s} {'Count':<8s} {'Percent':<10s}")
    print("-" * 40)
    for n in [5, 4, 3, 2, 1, 0]:
        c = counts.get(n, 0)
        pct = 100.0 * c / total
        print(f"{n}/5{'':<11s} {c:<8d} {pct:>6.1f}%")

    # ---- Per-session breakdown ----
    print("\n" + "=" * 60)
    print(" BY SESSION")
    print("=" * 60)
    for sess in ["cassette", "telemetry"]:
        sub = df[df["session"] == sess]
        if len(sub) == 0:
            continue
        print(f"\n{sess.upper()} (n={len(sub)}):")
        sub_counts = sub["aasm_pass"].value_counts().sort_index(ascending=False)
        for n in [5, 4, 3, 2, 1, 0]:
            c = sub_counts.get(n, 0)
            pct = 100.0 * c / len(sub)
            print(f"  {n}/5: {c:3d}  ({pct:5.1f}%)")

    # ---- Per-criterion pass rate ----
    print("\n" + "=" * 60)
    print(" PASS RATE PER CRITERION")
    print("=" * 60)
    for col in ["SE>85", "SOL<30", "WASO<20", "N3_13_23", "REM_20_25"]:
        n_pass = details_df[col].sum()
        pct = 100.0 * n_pass / total
        print(f"  {col:<12s}  {n_pass:3d}/{total}  ({pct:5.1f}%)")

    # ---- Correlations with SQS ----
    print("\n" + "=" * 60)
    print(" SQS vs AASM COMPLIANCE")
    print("=" * 60)
    from scipy.stats import spearmanr, pearsonr
    sp = spearmanr(df["sleep_quality_score"], df["aasm_pass"]).correlation
    pe = pearsonr(df["sleep_quality_score"], df["aasm_pass"])[0]
    print(f"  Spearman correlation: {sp:.4f}")
    print(f"  Pearson correlation:  {pe:.4f}")

    # Mean SQS per compliance level
    print("\n  Mean SQS by compliance level:")
    for n in [5, 4, 3, 2, 1, 0]:
        sub = df[df["aasm_pass"] == n]
        if len(sub) > 0:
            print(f"    {n}/5:  mean SQS = {sub['sleep_quality_score'].mean():.2f}  "
                  f"(n={len(sub)})")

    # Save
    df[["record_id", "subject_id", "session", "sleep_quality_score", "aasm_pass"]].to_csv(
        input_dir / "aasm_compliance.csv", index=False
    )
    details_df.to_csv(input_dir / "aasm_details.csv", index=False)
    print(f"\nSaved: {input_dir / 'aasm_compliance.csv'}")


if __name__ == "__main__":
    main()