# =============================================================================
# Sleep-EDF Expanded — Cassette → 10 sharded NPZ files
# -----------------------------------------------------------------------------
# Final cleaned version
#
# Output:
#   processed/cassette_shards/_global_meta.npz
#   processed/cassette_shards/shard_00.npz
#   ...
#   processed/cassette_shards/shard_09.npz
#   processed/cassette_shards/subject_to_shard.csv
#   processed/cassette_manifest.csv
#
# Key guarantees:
#   - Cassette recordings only
#   - All requested channels are retained
#   - 100 Hz, 30-second epochs => 3000 samples/epoch
#   - 5-class AASM labels: W, N1, N2, N3, REM
#   - Sleep stage 3 + 4 are merged into N3
#   - Subjects are kept entirely inside one shard
#   - Pass 1 and Pass 2 use exactly the same epoch-count logic
#   - Explicit zero-order-hold reconstruction for lower-rate channels
#   - Final verification checks shard integrity and class distribution
# =============================================================================

from __future__ import annotations

import gc
import re
import time
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import mne


# =============================================================================
# 1) CONFIGURATION
# =============================================================================

# --- Paths -------------------------------------------------------------------

DATA_ROOT: Path = Path(r"D:\Hackaton\sleep-edf-database-expanded-1.0.0")
CASSETTE_DIR: Path = DATA_ROOT / "sleep-cassette"
SUBJECTS_XLS: Path = DATA_ROOT / "SC-subjects.xls"

PROCESSED_DIR: Path = DATA_ROOT / "processed"
SHARD_DIR: Path = PROCESSED_DIR / "cassette_shards"
MANIFEST_CSV: Path = PROCESSED_DIR / "cassette_manifest.csv"

SHARD_DIR.mkdir(parents=True, exist_ok=True)


# --- Sharding ----------------------------------------------------------------

N_SHARDS: int = 10
RANDOM_SEED: int = 42


# --- Channels ----------------------------------------------------------------
# Keep all requested Cassette channels.

CHANNELS_TO_KEEP: tuple[str, ...] = (
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
    "Resp oro-nasal",
    "EMG submental",
    "Temp rectal",
    "Event marker",
)


# --- Output dtype -------------------------------------------------------------

X_DTYPE = np.float32


# --- Compression --------------------------------------------------------------

# False = faster and larger
# True  = slower and smaller

COMPRESS_NPZ: bool = False


# --- Signal parameters --------------------------------------------------------

EEG_SAMPLING_HZ: float = 100.0
EPOCH_DURATION_SEC: float = 30.0
SAMPLES_PER_EPOCH: int = int(
    EEG_SAMPLING_HZ * EPOCH_DURATION_SEC
)


# --- Stage mapping ------------------------------------------------------------

STAGE_MAP: dict[str, str] = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

EXCLUDED_STAGES: frozenset[str] = frozenset({
    "Sleep stage ?",
    "Sleep stage M",
    "Movement time",
})

STAGE_ORDER: tuple[str, ...] = (
    "W",
    "N1",
    "N2",
    "N3",
    "REM",
)

STAGE_TO_INT: dict[str, int] = {
    stage: i for i, stage in enumerate(STAGE_ORDER)
}


# --- SC-subjects.xls ----------------------------------------------------------

COL_SUBJECT: str = "subject"
COL_NIGHT: str = "night"
COL_AGE: str = "age"
COL_SEX: str = "sex (F=1)"

SEX_MAP: dict = {
    1: "F",
    2: "M",
}

SEX_TO_INT: dict[str, int] = {
    "F": 0,
    "M": 1,
}

SUBJECT_OFFSET: int = 0


# --- Recording key ------------------------------------------------------------

KEY_RE = re.compile(r"^SC4(\d{2})(\d)[A-Z]0$")


# =============================================================================
# 2) WARNING FILTERS
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message=".*highpass filters.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*lowpass filters.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*Highpass cutoff.*",
)


# =============================================================================
# 3) EDF HELPERS
# =============================================================================

def _native_rates_from_raw(
    raw: mne.io.BaseRaw,
) -> list[float]:
    """
    Extract native sampling rate for every EDF channel.

    EDF channels can have different sampling frequencies.
    """
    extras = raw._raw_extras[0]

    n_samps = extras.get("n_samps")
    record_length = extras.get("record_length")

    if n_samps is None or record_length is None:
        return [
            float(raw.info["sfreq"])
        ] * len(raw.ch_names)

    n_samps = np.asarray(
        n_samps,
        dtype=float,
    )

    rec_len = float(
        np.max(
            np.asarray(
                record_length,
                dtype=float,
            )
        )
    )

    if rec_len <= 0:
        return [
            float(raw.info["sfreq"])
        ] * len(raw.ch_names)

    rates = (
        n_samps / rec_len
    ).tolist()

    if len(rates) != len(raw.ch_names):
        return [
            float(raw.info["sfreq"])
        ] * len(raw.ch_names)

    return rates


def _looks_like_zoh(
    sig: np.ndarray,
    ratio: int,
    tol: float = 1e-3,
) -> bool:
    """
    Check whether a signal already looks like zero-order-held data.
    """
    if ratio <= 1:
        return True

    n = len(sig) // ratio

    if n == 0:
        return True

    blocks = sig[
        : n * ratio
    ].reshape(n, ratio)

    deviations = np.abs(
        blocks - blocks[:, :1]
    ).max(axis=1)

    return bool(
        (deviations < tol).all()
    )


def reconstruct_zoh(
    sig: np.ndarray,
    native_rate: float,
    target_rate: float = EEG_SAMPLING_HZ,
) -> tuple[np.ndarray, int, bool]:
    """
    Force zero-order-hold upsampling using np.repeat.

    No interpolation is performed.
    """
    ratio_f = target_rate / native_rate

    if not np.isclose(
        ratio_f,
        round(ratio_f),
    ):
        raise ValueError(
            f"Non-integer upsampling ratio "
            f"{ratio_f} "
            f"({native_rate} -> {target_rate} Hz)"
        )

    ratio = int(
        round(ratio_f)
    )

    if ratio == 1:
        return (
            sig.astype(X_DTYPE),
            1,
            True,
        )

    already_zoh = _looks_like_zoh(
        sig,
        ratio,
    )

    n_full = len(sig) // ratio

    if n_full == 0:
        return (
            np.asarray([], dtype=X_DTYPE),
            ratio,
            already_zoh,
        )

    downsampled = sig[
        : n_full * ratio : ratio
    ]

    upsampled = np.repeat(
        downsampled,
        ratio,
    )

    if len(upsampled) < len(sig):
        pad = len(sig) - len(upsampled)

        upsampled = np.concatenate([
            upsampled,
            np.full(
                pad,
                downsampled[-1],
                dtype=upsampled.dtype,
            ),
        ])

    return (
        upsampled[:len(sig)].astype(
            X_DTYPE
        ),
        ratio,
        already_zoh,
    )


def _extract_patient_code_from_raw(
    raw: mne.io.BaseRaw,
) -> str | None:
    """
    Extract patient code from EDF header when available.
    """
    try:
        extras = raw._raw_extras[0]

        for key in (
            "patient_code",
            "patientcode",
            "subject_info",
        ):
            if key not in extras:
                continue

            val = extras[key]

            if isinstance(val, dict):
                return val.get("his_id")

            return (
                str(val)
                if val
                else None
            )

    except Exception:
        pass

    return None


def get_usable_epoch_count_from_header(
    psg_path: Path,
) -> tuple[int, int, float]:
    """
    Determine the exact number of usable 30-s epochs from the EDF header.

    IMPORTANT:
    This function uses the same 100-Hz sample-count logic as read_psg().
    Therefore Pass 1 and Pass 2 cannot disagree because of duration
    rounding.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        raw = mne.io.read_raw_edf(
            str(psg_path),
            preload=False,
            stim_channel=None,
            verbose=False,
        )

        if raw.n_times == 0:
            raw.close()
            return 0, 0, 0.0

        duration_sec = float(
            raw.times[-1]
        )

        n_samples_100hz = int(
            round(
                raw.times[-1]
                * EEG_SAMPLING_HZ
            )
        ) + 1

        n_samples_100hz = min(
            n_samples_100hz,
            raw.n_times,
        )

        n_epochs = (
            n_samples_100hz
            // SAMPLES_PER_EPOCH
        )

        usable_samples = (
            n_epochs
            * SAMPLES_PER_EPOCH
        )

        raw.close()

    return (
        n_epochs,
        usable_samples,
        duration_sec,
    )


# =============================================================================
# 4) FULL PSG READING
# =============================================================================

def read_psg(
    path: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Read one PSG and reconstruct all requested channels to 100 Hz.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        raw = mne.io.read_raw_edf(
            str(path),
            preload=True,
            stim_channel=None,
            verbose=False,
        )

    native_rates = _native_rates_from_raw(raw)

    rate_by_ch = dict(
        zip(
            raw.ch_names,
            native_rates,
        )
    )

    if raw.n_times:
        n_samples_100hz = int(
            round(
                raw.times[-1]
                * EEG_SAMPLING_HZ
            )
        ) + 1
    else:
        n_samples_100hz = 0

    n_samples_100hz = min(
        n_samples_100hz,
        raw.n_times,
    )

    data: dict[str, np.ndarray] = {}

    missing: list[str] = []

    zoh_ok: dict[str, bool] = {}
    ratios: dict[str, int] = {}

    for name in CHANNELS_TO_KEEP:

        if name not in raw.ch_names:
            missing.append(name)

            data[name] = np.full(
                n_samples_100hz,
                np.nan,
                dtype=X_DTYPE,
            )

            zoh_ok[name] = True
            ratios[name] = 1

            continue

        idx = raw.ch_names.index(name)

        sig = raw.get_data(
            picks=[idx]
        )[0]

        try:
            up, ratio, was_zoh = reconstruct_zoh(
                sig,
                rate_by_ch[name],
            )

        except ValueError:
            # Keep the original signal if the native rate cannot be
            # represented by an integer ZOH ratio.
            up = sig.astype(X_DTYPE)
            ratio = 1
            was_zoh = False

        data[name] = up[
            :n_samples_100hz
        ]

        zoh_ok[name] = was_zoh
        ratios[name] = ratio

    df = pd.DataFrame(data)

    df.index.name = "sample_idx"

    qc = {
        "channels_in_file": list(
            raw.ch_names
        ),
        "native_rates_hz": rate_by_ch,
        "ratios_used": ratios,
        "mne_was_zoh": zoh_ok,
        "missing_channels": missing,
        "duration_sec": (
            float(raw.times[-1])
            if raw.n_times
            else 0.0
        ),
        "patient_code_in_header":
            _extract_patient_code_from_raw(
                raw
            ),
    }

    raw.close()

    return df, qc


# =============================================================================
# 5) HYPNOGRAM
# =============================================================================

def find_hypnogram_for_psg(
    psg_path: Path,
) -> Path:
    """
    Find the unique Hypnogram EDF corresponding to one PSG.
    """
    key7 = psg_path.stem[:7]

    candidates = [
        p
        for p in psg_path.parent.glob(
            "*-Hypnogram.edf"
        )
        if p.stem[:7] == key7
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly 1 Hypnogram for "
            f"{psg_path.name}, found "
            f"{len(candidates)}: "
            f"{[c.name for c in candidates]}"
        )

    return candidates[0]


def validate_hypnogram_alignment(
    ann: mne.Annotations,
    psg_duration_sec: float,
    epoch_sec: float = EPOCH_DURATION_SEC,
    tol: float = 1e-3,
) -> list[str]:
    """
    Check whether hypnogram annotations align to the 30-s epoch grid.
    """
    issues: list[str] = []

    if len(ann) == 0:
        return ["empty hypnogram"]

    onsets = np.asarray(
        ann.onset
    )

    ends = (
        onsets
        + np.asarray(ann.duration)
    )

    bad_onsets = onsets[
        ~np.isclose(
            onsets % epoch_sec,
            0,
            atol=tol,
        )
    ]

    bad_ends = ends[
        ~np.isclose(
            ends % epoch_sec,
            0,
            atol=tol,
        )
    ]

    for value in bad_onsets[:5]:
        issues.append(
            f"onset not aligned to "
            f"{epoch_sec:.0f}s grid: "
            f"{value}"
        )

    for value in bad_ends[:5]:
        issues.append(
            f"end not aligned to "
            f"{epoch_sec:.0f}s grid: "
            f"{value}"
        )

    if onsets.min() < -tol:
        issues.append(
            f"negative onset: "
            f"{onsets.min()}"
        )

    if ends.max() > (
        psg_duration_sec + tol
    ):
        issues.append(
            f"annotation end "
            f"{ends.max():.3f}s > PSG "
            f"duration "
            f"{psg_duration_sec:.3f}s"
        )

    return issues


def read_hypnogram_stages(
    hypno_path: Path,
    n_epochs: int,
    epoch_sec: float = EPOCH_DURATION_SEC,
) -> tuple[pd.Series, dict]:
    """
    Convert annotations to one stage label per 30-s epoch.
    """
    ann = mne.read_annotations(
        str(hypno_path)
    )

    labels = np.full(
        n_epochs,
        np.nan,
        dtype=object,
    )

    n_excluded_epochs = 0
    n_unmapped = 0

    onsets = np.asarray(
        ann.onset
    )

    durations = np.asarray(
        ann.duration
    )

    descriptions = np.asarray(
        ann.description
    )

    for start, dur, desc in zip(
        onsets,
        durations,
        descriptions,
    ):

        if desc in EXCLUDED_STAGES:
            n_excluded_epochs += int(
                round(
                    dur / epoch_sec
                )
            )
            continue

        if desc not in STAGE_MAP:
            n_unmapped += 1
            continue

        start_epoch = int(
            round(
                start / epoch_sec
            )
        )

        end_epoch = min(
            int(
                round(
                    (start + dur)
                    / epoch_sec
                )
            ),
            n_epochs,
        )

        if end_epoch > start_epoch:
            labels[
                start_epoch:end_epoch
            ] = STAGE_MAP[desc]

    series = pd.Series(
        labels,
        name="stage",
    )

    qc = {
        "n_annotations": len(ann),
        "n_excluded_epochs":
            n_excluded_epochs,
        "n_unmapped_annotations":
            n_unmapped,
        "n_unlabeled_epochs":
            int(series.isna().sum()),
        "first_onset_sec": (
            float(onsets.min())
            if len(ann)
            else float("nan")
        ),
        "last_end_sec": (
            float(
                (
                    onsets + durations
                ).max()
            )
            if len(ann)
            else float("nan")
        ),
    }

    return series, qc


# =============================================================================
# 6) SUBJECT METADATA
# =============================================================================

def load_subject_metadata(
    xls_path: Path,
    col_subject: str,
    col_night: str,
    col_age: str,
    col_sex: str,
    sex_map: dict,
    subject_offset: int = 0,
) -> pd.DataFrame:
    """
    Load and normalize SC-subjects.xls.
    """
    raw = pd.read_excel(
        xls_path
    )

    out = pd.DataFrame({
        "subject":
            raw[col_subject].astype(int)
            + subject_offset,

        "night":
            raw[col_night].astype(int),

        "age":
            raw[col_age].astype(int),

        "sex":
            raw[col_sex].map(sex_map),
    })

    bad = out["sex"].isna()

    if bad.any():
        raise ValueError(
            "Unmapped sex values: "
            f"{raw.loc[bad, col_sex].unique()}"
        )

    assert set(out["sex"]) <= {
        "F",
        "M",
    }

    return out.set_index(
        ["subject", "night"]
    )


# =============================================================================
# 7) RECORDING HELPERS
# =============================================================================

def parse_recording_key(
    recording_key: str,
) -> tuple[int, int]:
    """
    Parse SC4xxN?0 recording key.
    """
    match = KEY_RE.match(
        recording_key
    )

    if not match:
        raise ValueError(
            f"Unexpected recording key: "
            f"{recording_key}"
        )

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def measure_recording(
    psg_path: Path,
) -> tuple[int, int]:
    """
    Pass 1.

    Compute total and labeled epochs using EXACTLY the same EDF
    sample-count rule used during full processing.
    """
    (
        n_epochs,
        _,
        _,
    ) = get_usable_epoch_count_from_header(
        psg_path
    )

    if n_epochs <= 0:
        return 0, 0

    hyp_path = find_hypnogram_for_psg(
        psg_path
    )

    _, qc = read_hypnogram_stages(
        hyp_path,
        n_epochs,
    )

    n_kept = (
        n_epochs
        - qc["n_unlabeled_epochs"]
    )

    return (
        n_epochs,
        n_kept,
    )


def process_cassette_recording(
    psg_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """
    Pass 2.

    Process one complete Cassette PSG + Hypnogram.
    """
    signal_df, sig_qc = read_psg(
        psg_path
    )

    hyp_path = find_hypnogram_for_psg(
        psg_path
    )

    ann = mne.read_annotations(
        str(hyp_path)
    )

    grid_issues = (
        validate_hypnogram_alignment(
            ann,
            sig_qc["duration_sec"],
        )
    )

    n_epochs = (
        len(signal_df)
        // SAMPLES_PER_EPOCH
    )

    # -------------------------------------------------------------------------
    # Critical Pass-1 / Pass-2 consistency check
    # -------------------------------------------------------------------------

    (
        expected_n_epochs,
        _,
        _,
    ) = get_usable_epoch_count_from_header(
        psg_path
    )

    if n_epochs != expected_n_epochs:
        raise RuntimeError(
            f"Epoch-count mismatch for "
            f"{psg_path.name}: "
            f"Pass1={expected_n_epochs}, "
            f"Pass2={n_epochs}"
        )

    usable = (
        n_epochs
        * SAMPLES_PER_EPOCH
    )

    signal_df = signal_df.iloc[
        :usable
    ]

    stage_series, stage_qc = (
        read_hypnogram_stages(
            hyp_path,
            n_epochs,
        )
    )

    # -------------------------------------------------------------------------
    # Verify labeled runs occupy complete 30-s epochs
    # -------------------------------------------------------------------------

    stage_arr_all = np.repeat(
        stage_series.to_numpy(),
        SAMPLES_PER_EPOCH,
    )

    labeled = pd.notna(
        stage_arr_all
    )

    if labeled.any():

        changes = np.flatnonzero(
            np.diff(
                labeled.astype(
                    np.int8
                )
            ) != 0
        )

        segments = np.split(
            np.arange(
                len(labeled)
            ),
            changes + 1,
        )

        for seg in segments:

            if (
                labeled[seg[0]]
                and len(seg)
                % SAMPLES_PER_EPOCH
                != 0
            ):
                raise AssertionError(
                    f"Labeled run length "
                    f"{len(seg)} is not a "
                    f"multiple of "
                    f"{SAMPLES_PER_EPOCH} "
                    f"in {psg_path.name}"
                )

    # -------------------------------------------------------------------------
    # Convert to:
    # (epochs, channels, samples)
    # -------------------------------------------------------------------------

    arr = signal_df[
        list(CHANNELS_TO_KEEP)
    ].to_numpy(
        dtype=X_DTYPE
    )

    arr = arr.reshape(
        n_epochs,
        SAMPLES_PER_EPOCH,
        len(CHANNELS_TO_KEEP),
    )

    arr = np.transpose(
        arr,
        (0, 2, 1),
    ).astype(
        X_DTYPE,
        copy=False,
    )

    # -------------------------------------------------------------------------
    # Keep only labeled epochs
    # -------------------------------------------------------------------------

    stage_per_epoch = (
        stage_series.to_numpy()
    )

    keep = pd.notna(
        stage_per_epoch
    )

    X_epochs = arr[keep]

    y_epochs = np.array(
        [
            STAGE_TO_INT[str(stage)]
            for stage in
            stage_per_epoch[keep]
        ],
        dtype=np.int8,
    )

    epoch_ids = np.flatnonzero(
        keep
    ).astype(
        np.int32
    )

    # -------------------------------------------------------------------------
    # Recording metadata
    # -------------------------------------------------------------------------

    recording_key = (
        psg_path.stem
        .replace("-PSG", "")
    )

    subject, night = (
        parse_recording_key(
            recording_key
        )
    )

    qc = {
        "recording_key":
            recording_key,

        "psg_path":
            str(psg_path),

        "hypnogram_path":
            str(hyp_path),

        "subject":
            subject,

        "night":
            night,

        "duration_sec":
            sig_qc["duration_sec"],

        "n_epochs_total":
            n_epochs,

        "n_epochs_kept":
            int(keep.sum()),

        "missing_channels":
            sig_qc["missing_channels"],

        "mne_was_zoh":
            sig_qc["mne_was_zoh"],

        "patient_code_in_header":
            sig_qc[
                "patient_code_in_header"
            ],

        "grid_issues":
            grid_issues,

        **stage_qc,
    }

    # -------------------------------------------------------------------------
    # Free temporary arrays
    # -------------------------------------------------------------------------

    del (
        signal_df,
        arr,
        stage_arr_all,
        labeled,
    )

    return (
        X_epochs,
        y_epochs,
        epoch_ids,
        qc,
    )


def iter_cassette_psg(
    cassette_dir: Path,
) -> Iterator[Path]:
    """
    Yield Cassette PSG files only.
    """
    for path in sorted(
        cassette_dir.glob(
            "*-PSG.edf"
        )
    ):
        if path.stem.startswith("SC"):
            yield path


# =============================================================================
# 8) SUBJECT → SHARD ASSIGNMENT
# =============================================================================

def assign_subjects_to_shards(
    psg_paths: list[Path],
    n_shards: int,
    seed: int,
) -> tuple[
    dict[int, int],
    list[list[dict]],
]:
    """
    Randomly distribute subjects across shards.

    Both nights of a subject always remain in the same shard.
    """
    rec_meta = []

    for psg in psg_paths:

        key = (
            psg.stem
            .replace("-PSG", "")
        )

        subject, night = (
            parse_recording_key(key)
        )

        rec_meta.append({
            "psg_path": psg,
            "recording_key": key,
            "subject": subject,
            "night": night,
        })

    unique_subjects = np.array(
        sorted({
            item["subject"]
            for item in rec_meta
        })
    )

    rng = np.random.default_rng(
        seed
    )

    rng.shuffle(
        unique_subjects
    )

    groups = np.array_split(
        unique_subjects,
        n_shards,
    )

    subject_to_shard: dict[
        int, int
    ] = {}

    for shard_id, group in enumerate(
        groups
    ):
        for subject in group.tolist():
            subject_to_shard[
                int(subject)
            ] = shard_id

    per_shard_recordings = [
        []
        for _ in range(n_shards)
    ]

    for item in rec_meta:
        shard_id = (
            subject_to_shard[
                item["subject"]
            ]
        )

        per_shard_recordings[
            shard_id
        ].append(item)

    return (
        subject_to_shard,
        per_shard_recordings,
    )


# =============================================================================
# 9) TWO-PASS SHARDED PROCESSING
# =============================================================================

def process_all_to_shards(
    cassette_dir: Path,
    subject_meta: pd.DataFrame,
    shard_dir: Path,
    n_shards: int = N_SHARDS,
    seed: int = RANDOM_SEED,
) -> tuple[
    list[dict],
    list[dict],
    dict[int, int],
]:
    """
    Process all Cassette recordings into subject-safe NPZ shards.
    """
    psg_paths = list(
        iter_cassette_psg(
            cassette_dir
        )
    )

    print(
        f"Found {len(psg_paths)} "
        f"Cassette PSG files"
    )

    if not psg_paths:
        raise RuntimeError(
            "No Cassette PSG files found."
        )

    subject_to_shard, per_shard = (
        assign_subjects_to_shards(
            psg_paths,
            n_shards,
            seed,
        )
    )

    print(
        f"Assigned "
        f"{len(subject_to_shard)} "
        f"subjects into "
        f"{n_shards} shards"
    )

    for shard_id, recs in enumerate(
        per_shard
    ):
        n_subjects = len({
            r["subject"]
            for r in recs
        })

        print(
            f"  shard {shard_id:02d}: "
            f"{n_subjects:2d} subjects, "
            f"{len(recs):2d} recordings"
        )

    qc_rows: list[dict] = []
    issues: list[dict] = []

    t_start = time.time()

    # =========================================================================
    # SHARD LOOP
    # =========================================================================

    for shard_id, recs in enumerate(
        per_shard
    ):
        if not recs:
            continue

        print(
            f"\n=== Shard "
            f"{shard_id:02d} "
            f"({len(recs)} recordings) ==="
        )

        # =====================================================================
        # PASS 1 — exact sizing
        # =====================================================================

        sizes: list[
            tuple[int, int]
        ] = []

        for r in recs:

            try:
                n_total, n_kept = (
                    measure_recording(
                        r["psg_path"]
                    )
                )

            except Exception as exc:

                print(
                    f"  MEASURE ERROR "
                    f"[{r['recording_key']}]: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                issues.append({
                    "recording_key":
                        r["recording_key"],

                    "error":
                        f"{type(exc).__name__}: "
                        f"{exc}",
                })

                sizes.append(
                    (0, 0)
                )

                continue

            sizes.append(
                (n_total, n_kept)
            )

        n_shard_total = sum(
            n_kept
            for _, n_kept in sizes
        )

        estimated_gb = (
            n_shard_total
            * len(CHANNELS_TO_KEEP)
            * SAMPLES_PER_EPOCH
            * np.dtype(X_DTYPE).itemsize
            / 1e9
        )

        print(
            f"  Shard will hold "
            f"{n_shard_total:,} "
            f"labeled epochs "
            f"({estimated_gb:.2f} GB)"
        )

        if n_shard_total == 0:
            print(
                "  No labeled epochs. "
                "Skipping shard."
            )
            continue

        # =====================================================================
        # PRE-ALLOCATE
        # =====================================================================

        n_channels = len(
            CHANNELS_TO_KEEP
        )

        X = np.empty(
            (
                n_shard_total,
                n_channels,
                SAMPLES_PER_EPOCH,
            ),
            dtype=X_DTYPE,
        )

        y = np.empty(
            (n_shard_total,),
            dtype=np.int8,
        )

        age_arr = np.empty(
            (n_shard_total,),
            dtype=np.int16,
        )

        sex_arr = np.empty(
            (n_shard_total,),
            dtype=np.int8,
        )

        subj_arr = np.empty(
            (n_shard_total,),
            dtype=np.int16,
        )

        night_arr = np.empty(
            (n_shard_total,),
            dtype=np.int8,
        )

        ep_id_arr = np.empty(
            (n_shard_total,),
            dtype=np.int32,
        )

        rec_key_arr = np.empty(
            (n_shard_total,),
            dtype="U10",
        )

        # =====================================================================
        # PASS 2 — actual processing
        # =====================================================================

        pos = 0

        for rec_idx, r in enumerate(
            recs
        ):

            expected_total, expected_kept = (
                sizes[rec_idx]
            )

            # If Pass 1 failed, skip.
            if expected_kept == 0:
                continue

            try:
                (
                    X_ep,
                    y_ep,
                    ep_ids,
                    qc,
                ) = process_cassette_recording(
                    r["psg_path"]
                )

            except Exception as exc:

                print(
                    f"  PROCESS ERROR "
                    f"[{r['recording_key']}]: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                issues.append({
                    "recording_key":
                        r["recording_key"],

                    "error":
                        f"{type(exc).__name__}: "
                        f"{exc}",
                })

                continue

            n = len(y_ep)

            # -----------------------------------------------------------------
            # CRITICAL INTEGRITY CHECK
            # -----------------------------------------------------------------

            if n != expected_kept:
                raise RuntimeError(
                    f"Pass-1 / Pass-2 mismatch "
                    f"for {r['recording_key']}: "
                    f"Pass1 kept={expected_kept}, "
                    f"Pass2 kept={n}, "
                    f"Pass1 total={expected_total}"
                )

            if n == 0:
                continue

            subject = qc["subject"]
            night = qc["night"]

            # -----------------------------------------------------------------
            # Metadata merge
            # -----------------------------------------------------------------

            try:
                metadata = subject_meta.loc[
                    (subject, night)
                ]

                age_val = int(
                    metadata["age"]
                )

                sex_val = str(
                    metadata["sex"]
                )

            except KeyError:

                age_val = -1
                sex_val = "F"

                issues.append({
                    "recording_key":
                        qc["recording_key"],

                    "error":
                        "No spreadsheet row for "
                        f"(subject={subject}, "
                        f"night={night})",
                })

            # -----------------------------------------------------------------
            # EDF header cross-check
            # -----------------------------------------------------------------

            header_code = qc.get(
                "patient_code_in_header"
            )

            if header_code:

                normalized = (
                    str(header_code)
                    .strip()
                    .split()[0]
                )

                if (
                    normalized
                    and normalized[:7]
                    != qc["recording_key"][:7]
                ):
                    issues.append({
                        "recording_key":
                            qc["recording_key"],

                        "warning":
                            f"Header patient_code "
                            f"'{header_code}' differs "
                            f"from filename key "
                            f"'{qc['recording_key']}'",
                    })

            # -----------------------------------------------------------------
            # Write
            # -----------------------------------------------------------------

            end = pos + n

            X[pos:end] = X_ep
            y[pos:end] = y_ep

            age_arr[pos:end] = age_val
            sex_arr[pos:end] = (
                SEX_TO_INT[sex_val]
            )

            subj_arr[pos:end] = subject
            night_arr[pos:end] = night

            ep_id_arr[pos:end] = ep_ids

            rec_key_arr[pos:end] = (
                qc["recording_key"]
            )

            pos = end

            # -----------------------------------------------------------------
            # QC row
            # -----------------------------------------------------------------

            qc_rows.append({
                "recording_key":
                    qc["recording_key"],

                "shard_id":
                    shard_id,

                "subject":
                    subject,

                "night":
                    night,

                "age":
                    age_val,

                "sex":
                    sex_val,

                "n_epochs_total":
                    qc["n_epochs_total"],

                "n_epochs_kept":
                    qc["n_epochs_kept"],

                "n_excluded_epochs":
                    qc["n_excluded_epochs"],

                "n_unmapped_annotations":
                    qc["n_unmapped_annotations"],

                "missing_channels":
                    ",".join(
                        qc["missing_channels"]
                    ),

                "grid_issue_count":
                    len(
                        qc["grid_issues"]
                    ),
            })

            print(
                f"    + "
                f"{qc['recording_key']:>10s}  "
                f"epochs={n:>4d}  "
                f"age={age_val}  "
                f"sex={sex_val}"
            )

            del (
                X_ep,
                y_ep,
                ep_ids,
            )

            gc.collect()

        # =====================================================================
        # FINAL SHARD INTEGRITY CHECK
        # =====================================================================

        if pos != n_shard_total:
            raise RuntimeError(
                f"Shard {shard_id:02d} incomplete: "
                f"allocated={n_shard_total:,}, "
                f"written={pos:,}"
            )

        # =====================================================================
        # SAVE
        # =====================================================================

        shard_path = (
            shard_dir
            / f"shard_{shard_id:02d}.npz"
        )

        print(
            f"  Saving "
            f"{shard_path.name} ..."
        )

        t0 = time.time()

        saver = (
            np.savez_compressed
            if COMPRESS_NPZ
            else np.savez
        )

        saver(
            shard_path,
            X=X,
            y=y,
            age=age_arr,
            sex_int=sex_arr,
            subject=subj_arr,
            night=night_arr,
            epoch_id=ep_id_arr,
            recording_key=rec_key_arr,
        )

        dt = time.time() - t0

        size_gb = (
            shard_path.stat().st_size
            / 1e9
        )

        print(
            f"  Saved "
            f"{shard_path.name}: "
            f"{size_gb:.2f} GB "
            f"in {dt:.1f} s"
        )

        # =====================================================================
        # FREE
        # =====================================================================

        del (
            X,
            y,
            age_arr,
            sex_arr,
            subj_arr,
            night_arr,
            ep_id_arr,
            rec_key_arr,
        )

        gc.collect()

    # =========================================================================
    # GLOBAL METADATA
    # =========================================================================

    np.savez(
        shard_dir
        / "_global_meta.npz",

        channel_names=np.asarray(
            CHANNELS_TO_KEEP,
            dtype=str,
        ),

        label_names=np.asarray(
            STAGE_ORDER,
            dtype=str,
        ),

        sex_names=np.asarray(
            ["F", "M"],
            dtype=str,
        ),

        sfreq=np.asarray(
            int(EEG_SAMPLING_HZ)
        ),

        epoch_duration=np.asarray(
            int(EPOCH_DURATION_SEC)
        ),

        samples_per_epoch=np.asarray(
            int(SAMPLES_PER_EPOCH)
        ),

        n_channels=np.asarray(
            len(CHANNELS_TO_KEEP)
        ),

        n_shards=np.asarray(
            int(n_shards)
        ),

        random_seed=np.asarray(
            int(seed)
        ),
    )

    # =========================================================================
    # SUBJECT → SHARD CSV
    # =========================================================================

    subj_shard_df = pd.DataFrame(
        sorted(
            subject_to_shard.items()
        ),
        columns=[
            "subject",
            "shard_id",
        ],
    )

    subj_shard_df.to_csv(
        shard_dir
        / "subject_to_shard.csv",
        index=False,
    )

    print(
        f"\nTotal elapsed: "
        f"{time.time() - t_start:.1f} s"
    )

    return (
        qc_rows,
        issues,
        subject_to_shard,
    )


# =============================================================================
# 10) LOADING HELPERS
# =============================================================================

def load_shard(
    shard_path: Path,
) -> dict:
    """
    Load one shard into RAM.
    """
    with np.load(
        shard_path,
        allow_pickle=False,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }


def load_global_meta(
    shard_dir: Path,
) -> dict:
    """
    Load global metadata.
    """
    with np.load(
        shard_dir
        / "_global_meta.npz",
        allow_pickle=False,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }


def iter_shards(
    shard_dir: Path,
) -> Iterator[
    tuple[str, dict]
]:
    """
    Iterate through shards one at a time.
    """
    for path in sorted(
        shard_dir.glob(
            "shard_*.npz"
        )
    ):
        yield (
            path.stem,
            load_shard(path),
        )


def iter_batches(
    shard_dir: Path,
    batch_size: int = 256,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> Iterator[dict]:
    """
    Yield batches from one shard at a time.

    NOTE:
    One complete shard is loaded into RAM before batching.
    """
    if rng is None:
        rng = np.random.default_rng(
            RANDOM_SEED
        )

    shard_paths = sorted(
        shard_dir.glob(
            "shard_*.npz"
        )
    )

    if shuffle:
        rng.shuffle(
            shard_paths
        )

    for shard_path in shard_paths:

        data = load_shard(
            shard_path
        )

        n = len(
            data["y"]
        )

        indices = np.arange(
            n
        )

        if shuffle:
            rng.shuffle(
                indices
            )

        for start in range(
            0,
            n,
            batch_size,
        ):
            selected = indices[
                start:
                start + batch_size
            ]

            yield {
                key: value[selected]
                for key, value
                in data.items()
            }

        del data

        gc.collect()


# =============================================================================
# 11) FINAL VERIFICATION
# =============================================================================

def verify_shards(
    shard_dir: Path,
) -> None:
    """
    Verify shapes, metadata lengths, class labels and shard consistency.
    """
    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL SHARD INTEGRITY CHECK"
    )

    print(
        "=" * 72
    )

    meta = load_global_meta(
        shard_dir
    )

    n_channels = int(
        meta["n_channels"]
    )

    samples_per_epoch = int(
        meta["samples_per_epoch"]
    )

    expected_channels = len(
        CHANNELS_TO_KEEP
    )

    if n_channels != expected_channels:
        raise AssertionError(
            "Global metadata channel count "
            "does not match configuration."
        )

    total_epochs = 0

    for shard_path in sorted(
        shard_dir.glob(
            "shard_*.npz"
        )
    ):
        data = load_shard(
            shard_path
        )

        n = len(
            data["y"]
        )

        expected_shape = (
            n,
            n_channels,
            samples_per_epoch,
        )

        if data["X"].shape != expected_shape:
            raise AssertionError(
                f"{shard_path.name}: "
                f"X shape "
                f"{data['X'].shape} != "
                f"{expected_shape}"
            )

        for key in (
            "y",
            "age",
            "sex_int",
            "subject",
            "night",
            "epoch_id",
            "recording_key",
        ):
            if len(data[key]) != n:
                raise AssertionError(
                    f"{shard_path.name}: "
                    f"{key} length mismatch."
                )

        unique_labels = np.unique(
            data["y"]
        )

        if not np.isin(
            unique_labels,
            np.arange(
                len(STAGE_ORDER)
            ),
        ).all():
            raise AssertionError(
                f"{shard_path.name}: "
                f"invalid labels "
                f"{unique_labels}"
            )

        total_epochs += n

        print(
            f"  {shard_path.name}: "
            f"{n:,} epochs | "
            f"shape={data['X'].shape}"
        )

        del data

    print(
        f"\nTotal verified epochs: "
        f"{total_epochs:,}"
    )

    print(
        "Integrity check: PASS"
    )


# =============================================================================
# 12) MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Basic path checks
    # -------------------------------------------------------------------------

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist:\n"
            f"{DATA_ROOT}"
        )

    if not CASSETTE_DIR.exists():
        raise FileNotFoundError(
            f"CASSETTE_DIR does not exist:\n"
            f"{CASSETTE_DIR}"
        )

    if not SUBJECTS_XLS.exists():
        raise FileNotFoundError(
            f"SC-subjects.xls does not exist:\n"
            f"{SUBJECTS_XLS}"
        )

    # -------------------------------------------------------------------------
    # Load subject metadata
    # -------------------------------------------------------------------------

    subject_meta = load_subject_metadata(
        SUBJECTS_XLS,
        COL_SUBJECT,
        COL_NIGHT,
        COL_AGE,
        COL_SEX,
        SEX_MAP,
        subject_offset=SUBJECT_OFFSET,
    )

    print(
        f"Rows in normalized metadata: "
        f"{len(subject_meta)}"
    )

    # -------------------------------------------------------------------------
    # Process Cassette → shards
    # -------------------------------------------------------------------------

    (
        qc_rows,
        issues,
        subject_to_shard,
    ) = process_all_to_shards(
        CASSETTE_DIR,
        subject_meta,
        SHARD_DIR,
        n_shards=N_SHARDS,
        seed=RANDOM_SEED,
    )

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------

    qc_df = pd.DataFrame(
        qc_rows
    )

    if not qc_df.empty:
        qc_df = (
            qc_df
            .sort_values(
                [
                    "shard_id",
                    "subject",
                    "night",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    qc_df.to_csv(
        MANIFEST_CSV,
        index=False,
    )

    print(
        f"\nManifest written to:\n"
        f"{MANIFEST_CSV}"
    )

    # -------------------------------------------------------------------------
    # Final verification
    # -------------------------------------------------------------------------

    verify_shards(
        SHARD_DIR
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL VERIFICATION REPORT"
    )

    print(
        "=" * 72
    )

    if qc_df.empty:
        print(
            "No recordings were successfully processed."
        )

        if issues:
            print(
                "\nIssues:"
            )

            for item in issues:
                print(
                    f"  {item}"
                )

        raise SystemExit(1)

    print(
        f"Recordings processed : "
        f"{len(qc_df)}"
    )

    print(
        f"Total labeled epochs : "
        f"{int(qc_df['n_epochs_kept'].sum()):,}"
    )

    print(
        f"Total excluded epochs: "
        f"{int(qc_df['n_excluded_epochs'].sum()):,}"
    )

    # =========================================================================
    # STREAMING CLASS STATS
    # =========================================================================

    class_counts = np.zeros(
        len(STAGE_ORDER),
        dtype=np.int64,
    )

    n_total = 0

    ages: set[int] = set()
    sexes: set[int] = set()

    shard_sizes = []

    for name, data in iter_shards(
        SHARD_DIR
    ):

        class_counts += np.bincount(
            data["y"],
            minlength=len(
                STAGE_ORDER
            ),
        )

        n_total += len(
            data["y"]
        )

        ages.update(
            np.unique(
                data["age"]
            ).tolist()
        )

        sexes.update(
            np.unique(
                data["sex_int"]
            ).tolist()
        )

        shard_sizes.append(
            (
                name,
                len(
                    data["y"]
                ),
            )
        )

        del data

    # =========================================================================
    # CLASS DISTRIBUTION
    # =========================================================================

    print(
        "\nOverall class distribution "
        "(epoch counts):"
    )

    for i, stage in enumerate(
        STAGE_ORDER
    ):

        count = int(
            class_counts[i]
        )

        percentage = (
            100 * count
            / max(n_total, 1)
        )

        print(
            f"  {stage:>4}: "
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )

    # =========================================================================
    # AGE / SEX
    # =========================================================================

    if ages:
        print(
            f"\nAge range: "
            f"{min(ages)} – "
            f"{max(ages)}"
        )

    print(
        f"Sex codes: "
        f"F={sum(x == 0 for x in sexes)} "
        f"M={sum(x == 1 for x in sexes)}"
    )

    # =========================================================================
    # SHARD SIZES
    # =========================================================================

    print(
        "\nShard sizes "
        "(labeled epochs):"
    )

    for name, n in shard_sizes:
        print(
            f"  {name}: "
            f"{n:>8,}"
        )

    # =========================================================================
    # RECORDING SUMMARY
    # =========================================================================

    print(
        "\nPer-recording summary "
        "(first 20 rows):"
    )

    columns = [
        "recording_key",
        "shard_id",
        "subject",
        "night",
        "age",
        "sex",
        "n_epochs_total",
        "n_epochs_kept",
        "n_excluded_epochs",
    ]

    print(
        qc_df[columns]
        .head(20)
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # MISSING CHANNELS
    # =========================================================================

    print(
        "\nRecordings with missing channels:"
    )

    miss = qc_df[
        qc_df["missing_channels"] != ""
    ]

    if miss.empty:
        print("  (none)")
    else:
        for _, row in miss.iterrows():
            print(
                f"  {row['recording_key']}: "
                f"{row['missing_channels']}"
            )

    # =========================================================================
    # ISSUES
    # =========================================================================

    print(
        "\nSpreadsheet / EDF mismatches "
        "and errors:"
    )

    if not issues:
        print("  (none)")
    else:
        for item in issues:
            print(
                f"  {item}"
            )

    # =========================================================================
    # GRID ISSUES
    # =========================================================================

    print(
        "\nGrid alignment issues "
        "(hypnogram not aligned to 30 s):"
    )

    grid = qc_df[
        qc_df["grid_issue_count"] > 0
    ]

    if grid.empty:
        print("  (none)")
    else:
        print(
            grid[
                [
                    "recording_key",
                    "grid_issue_count",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # MISSING NIGHT
    # =========================================================================

    print(
        "\nSubjects with only one recording "
        "(missing night):"
    )

    per_subject = (
        qc_df
        .groupby("subject")["night"]
        .nunique()
    )

    incomplete = per_subject[
        per_subject < 2
    ]

    if incomplete.empty:
        print("  (none)")
    else:
        for subject, n_recordings in (
            incomplete.items()
        ):
            nights = qc_df.loc[
                qc_df["subject"] == subject,
                "night",
            ].tolist()

            print(
                f"  subject={subject}, "
                f"recordings={n_recordings}, "
                f"nights={nights}"
            )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "=" * 72
    )
