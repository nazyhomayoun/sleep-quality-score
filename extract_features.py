"""
extract_features.py
===================
Extracts per-recording features from the 4-channel epoch-level signals
produced by build_sleep_dataset.py, using the sleep window defined in
sleep_quality.csv.

Input:
    processed_4ch/signals/*.npz         (X: n_epochs, 4, 3000)
    processed_4ch/sleep_quality.csv     (sleep_start_epoch, sleep_end_epoch)

Output:
    processed_4ch/features.csv          (one row per record)

Usage:
    python extract_features.py --input_dir "processed_4ch"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

# --- NumPy 2.x compatibility ---
# np.trapz was renamed to np.trapezoid in NumPy 2.0
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
# ------------------------------

# Channel order in npz files (must match --channels of build_sleep_dataset.py)
CH_EEG1 = 0  # EEG Fpz-Cz
CH_EEG2 = 1  # EEG Pz-Oz
CH_EOG  = 2  # EOG horizontal
CH_EMG  = 3  # EMG submental

SFREQ = 100.0  # Sleep-EDF Expanded

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}


# ----------------------------------------------------------------------
# Signal helpers
# ----------------------------------------------------------------------
def _bandpower(freqs, psd, band):
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _hjorth_params(sig):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    var0 = float(np.var(sig))
    var1 = float(np.var(d1))
    var2 = float(np.var(d2))
    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 1e-12 else 0.0
    complexity = (np.sqrt(var2 / var1) / mobility) if (var1 > 1e-12 and mobility > 1e-12) else 0.0
    return activity, float(mobility), float(complexity)


def _spectral_entropy(freqs, psd):
    p = np.asarray(psd, dtype=float)
    p = p / (p.sum() + 1e-12)
    p = p[p > 1e-12]
    if len(p) <= 1:
        return 0.0
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(len(p)))


def _welch(sig):
    freqs, psd = signal.welch(sig, fs=SFREQ, nperseg=512, noverlap=256)
    return freqs, psd


# ----------------------------------------------------------------------
# Per-channel feature extraction
# ----------------------------------------------------------------------
def extract_eeg_features(sig_full, prefix):
    feats = {}
    freqs, psd = _welch(sig_full)

    total_power = np.trapz(psd, freqs) + 1e-12
    band_powers = {}
    for name, band in BANDS.items():
        bp = _bandpower(freqs, psd, band)
        band_powers[name] = bp
        feats[f"{prefix}_{name}_abs"] = float(bp)
        feats[f"{prefix}_{name}_rel"] = float(bp / total_power)

    feats[f"{prefix}_delta_beta_ratio"] = float(
        band_powers["delta"] / (band_powers["beta"] + 1e-12))
    feats[f"{prefix}_theta_alpha_ratio"] = float(
        band_powers["theta"] / (band_powers["alpha"] + 1e-12))
    feats[f"{prefix}_delta_theta_ratio"] = float(
        band_powers["delta"] / (band_powers["theta"] + 1e-12))
    feats[f"{prefix}_slow_fast_ratio"] = float(
        (band_powers["delta"] + band_powers["theta"]) /
        (band_powers["alpha"] + band_powers["beta"] + 1e-12))

    feats[f"{prefix}_spec_entropy"] = _spectral_entropy(freqs, psd)

    n_epochs = len(sig_full) // 3000
    acts, mobs, cmps = [], [], []
    for i in range(n_epochs):
        seg = sig_full[i * 3000:(i + 1) * 3000]
        a, m, c = _hjorth_params(seg)
        acts.append(a); mobs.append(m); cmps.append(c)

    feats[f"{prefix}_hjorth_activity_mean"] = float(np.mean(acts)) if acts else 0.0
    feats[f"{prefix}_hjorth_activity_std"]  = float(np.std(acts))  if acts else 0.0
    feats[f"{prefix}_hjorth_mobility_mean"] = float(np.mean(mobs)) if mobs else 0.0
    feats[f"{prefix}_hjorth_mobility_std"]  = float(np.std(mobs))  if mobs else 0.0
    feats[f"{prefix}_hjorth_complexity_mean"] = float(np.mean(cmps)) if cmps else 0.0
    feats[f"{prefix}_hjorth_complexity_std"]  = float(np.std(cmps))  if cmps else 0.0

    return feats


def extract_eog_features(sig_full):
    feats = {}
    feats["eog_rms"] = float(np.sqrt(np.mean(sig_full ** 2)))
    feats["eog_variance"] = float(np.var(sig_full))
    zc = np.sum(np.abs(np.diff(np.sign(sig_full))) > 0)
    feats["eog_zero_crossings_per_s"] = float(zc / (len(sig_full) / SFREQ))
    a, m, c = _hjorth_params(sig_full)
    feats["eog_hjorth_activity"] = float(a)
    feats["eog_hjorth_mobility"] = float(m)
    return feats


def extract_emg_features(sig_full):
    feats = {}
    feats["emg_rms"] = float(np.sqrt(np.mean(sig_full ** 2)))
    feats["emg_variance"] = float(np.var(sig_full))
    freqs, psd = _welch(sig_full)
    hf_mask = (freqs >= 20.0) & (freqs < 50.0)
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
    total = np.trapz(psd, freqs) + 1e-12
    feats["emg_high_freq_power"] = hf_power
    feats["emg_hf_rel"] = float(hf_power / total)
    a, m, c = _hjorth_params(sig_full)
    feats["emg_hjorth_activity"] = float(a)
    feats["emg_hjorth_mobility"] = float(m)
    return feats


# ----------------------------------------------------------------------
# Per-record pipeline
# ----------------------------------------------------------------------
def extract_features_for_record(X, sleep_start, sleep_end):
    if sleep_start < 0 or sleep_end < 0 or sleep_end <= sleep_start:
        return None
    window = X[sleep_start:sleep_end + 1]        # (n_sleep_epochs, 4, 3000)
    if window.shape[0] < 1:
        return None
    n_samples = window.shape[0] * 3000
    if n_samples < 2048:
        return None

    feats = {
        "n_sleep_epochs": int(window.shape[0]),
        "n_samples_in_window": int(n_samples),
    }

    eeg1 = window[:, CH_EEG1, :].reshape(-1).astype(np.float64)
    eeg2 = window[:, CH_EEG2, :].reshape(-1).astype(np.float64)
    eog  = window[:, CH_EOG,  :].reshape(-1).astype(np.float64)
    emg  = window[:, CH_EMG,  :].reshape(-1).astype(np.float64)

    feats.update(extract_eeg_features(eeg1, "eeg1"))
    feats.update(extract_eeg_features(eeg2, "eeg2"))
    feats.update(extract_eog_features(eog))
    feats.update(extract_emg_features(emg))

    return feats


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract features per recording.")
    parser.add_argument("--input_dir", type=str, default="processed_4ch",
                         help="Folder containing signals/ and sleep_quality.csv")
    parser.add_argument("--output_csv", type=str, default=None,
                         help="Output CSV (default: <input_dir>/features.csv)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    signals_dir = input_dir / "signals"
    sq_path = input_dir / "sleep_quality.csv"
    if not sq_path.exists():
        raise FileNotFoundError(f"sleep_quality.csv not found: {sq_path}")
    if not signals_dir.exists():
        raise FileNotFoundError(f"signals folder not found: {signals_dir}")

    output_csv = Path(args.output_csv) if args.output_csv else input_dir / "features.csv"

    sq = pd.read_csv(sq_path)
    print(f"Records in sleep_quality.csv: {len(sq)}")

    rows = []
    n_total = len(sq)
    n_ok, n_skip = 0, 0
    for i, r in enumerate(sq.itertuples(index=False), start=1):
        rec = r.record_id
        npz_path = signals_dir / f"{rec}.npz"
        if not npz_path.exists():
            print(f"[{i}/{n_total}] ⚠️  missing {npz_path.name}")
            n_skip += 1
            continue

        d = np.load(npz_path, allow_pickle=True)
        X = d["X"]

        feats = extract_features_for_record(
            X,
            int(getattr(r, "sleep_start_epoch", -1)),
            int(getattr(r, "sleep_end_epoch", -1)),
        )
        if feats is None:
            print(f"[{i}/{n_total}] ⚠️  no valid sleep window for {rec}")
            n_skip += 1
            continue

        row = {
            "record_id": rec,
            "subject_id": r.subject_id,
            "night": int(r.night),
            "session": r.session,
        }
        row.update(feats)
        rows.append(row)
        n_ok += 1

        if i % 20 == 0 or i == n_total:
            print(f"[{i}/{n_total}] {rec}: {len(feats)} features")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\n========== Summary ==========")
    print(f"Records processed: {n_ok}  |  skipped: {n_skip}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Feature columns: {df.shape[1] - 4}  (excluding record_id, subject_id, night, session)")
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    main()