"""
build_sleep_dataset.py
=======================
این اسکریپت فایل‌های PSG.edf و Hypnogram.edf دیتاست Sleep-EDF Expanded را می‌خواند،
سیگنال را به epoch های ۳۰ ثانیه‌ای (استاندارد امتیازدهی خواب) تقسیم می‌کند،
برچسب مرحله خواب را به هر epoch می‌چسباند، و خروجی را به شکل زیر ذخیره می‌کند:

    output_dir/
    ├── metadata.csv          <- یک ردیف به ازای هر epoch (شامل subject_id, night, ...)
    └── signals/
        ├── SC4001E0.npz      <- آرایه سیگنال تمام epoch های این رکورد
        ├── SC4002E0.npz
        └── ...

هدف: نگه‌داشتن subject_id در متادیتا تا بعداً بتوانید split گروهی (per-subject) بزنید
و از نشت داده (data leakage) بین train/test جلوگیری کنید.

نحوه اجرا:
    python build_sleep_dataset.py \
        --data_dir "sleep-edf-database-expanded-1.0.0" \
        --output_dir "processed" \
        --epoch_sec 30 \
        --channels "EEG Fpz-Cz" "EEG Pz-Oz" \
        --crop_wake_minutes 30
"""

import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import mne

mne.set_log_level("ERROR")  # لاگ‌های پرحجم mne را خاموش می‌کنیم


# ----------------------------------------------------------------------
# نگاشت برچسب‌های خام Hypnogram به کدهای عددی استاندارد
# مراحل 3 و 4 (R&K) را طبق رویه رایج در N3 ادغام می‌کنیم (مطابق AASM)
# ----------------------------------------------------------------------
STAGE_TO_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage R": 4,
    # مراحلی که کنار می‌گذاریم (نه بخشی از خواب معتبر برای مدل‌سازی)
    "Sleep stage ?": None,
    "Movement time": None,
}
ID_TO_STAGE_NAME = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# ----------------------------------------------------------------------
# 1) پیدا کردن جفت‌های PSG + Hypnogram و استخراج subject_id از نام فایل
# ----------------------------------------------------------------------
def parse_filename(psg_path: Path):
    """
    از روی نام فایل، session (SC/ST)، شماره فرد (subject_id) و شماره شب (night)
    را استخراج می‌کند.

    الگوی نام‌گذاری:
      SC4ssNEO-PSG.edf   -> ss = شماره فرد (۲ رقم), N = شماره شب (۱ رقم)
      ST7ssNJ0-PSG.edf   -> همین ساختار برای مطالعه Telemetry
    """
    stem = psg_path.stem  # مثلا "SC4001E0-PSG" -> بدون پسوند
    name = stem.replace("-PSG", "")

    m = re.match(r"^(SC|ST)([47])(\d{2})(\d)", name)
    if not m:
        raise ValueError(f"نام فایل با الگوی شناخته‌شده مطابقت ندارد: {psg_path.name}")

    session_prefix, _, subj_num, night = m.groups()
    session = "cassette" if session_prefix == "SC" else "telemetry"
    subject_id = f"{session_prefix}4{subj_num}" if session == "cassette" else f"{session_prefix}7{subj_num}"

    return {
        "record_id": name,          # شناسه یکتای این ضبط خاص، مثلا SC4001E0
        "subject_id": subject_id,   # شناسه فرد که بین دو شب او مشترک است -> برای split
        "night": int(night),
        "session": session,
    }


def find_pairs(data_dir: Path):
    """
    تمام فایل‌های *_PSG.edf را پیدا می‌کند و فایل Hypnogram متناظرشان را
    (که ۷ کاراکتر اول نامشان با PSG یکی است ولی حرف هشتم فرق دارد) جفت می‌کند.
    """
    pairs = []
    for sub_dir_name in ["sleep-cassette", "sleep-telemetry"]:
        sub_dir = data_dir / sub_dir_name
        if not sub_dir.exists():
            continue
        for psg_path in sorted(sub_dir.glob("*-PSG.edf")):
            base = psg_path.stem[:7]  # مثلا "SC4001E" ، بدون کاراکتر آخر (E0 -> E)
            hyp_candidates = list(sub_dir.glob(f"{base}*-Hypnogram.edf"))
            if not hyp_candidates:
                print(f"⚠️  Hypnogram پیدا نشد برای: {psg_path.name} -- رد شد")
                continue
            pairs.append((psg_path, hyp_candidates[0]))
    return pairs


# ----------------------------------------------------------------------
# 2) خواندن یک رکورد و تبدیل آن به epoch های برچسب‌گذاری‌شده
# ----------------------------------------------------------------------
def extract_epochs_for_record(psg_path: Path, hyp_path: Path, epoch_sec: int,
                               channels: list, crop_wake_minutes: float):
    """
    Read one PSG + Hypnogram file and return the 30-second epoch array and
    corresponding labels.

    Returns:
        X : np.ndarray of shape (n_epochs, n_channels, n_samples_per_epoch)
        y : np.ndarray of shape (n_epochs,)  -- numeric sleep stage code
        onsets : np.ndarray  -- start time of each epoch in seconds from the beginning of the recording
    """
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    # Keep only the requested channels (if present in this record)
    available = [ch for ch in channels if ch in raw.ch_names]
    if not available:
        raise ValueError(f"None of the channels {channels} were found in {psg_path.name}. "
                          f"Available channels: {raw.ch_names}")
    raw.pick(available)

    annot = mne.read_annotations(hyp_path)

    # ---- Clean annotations BEFORE handing them to MNE ----
    # 1) Rename "Sleep stage 4" to "Sleep stage 3" so we have a single N3 label.
    # 2) Drop annotations we do not want ("Sleep stage ?", "Movement time", and
    #    the very first "Sleep stage W" placeholder at t=0 if present).
    KEEP_DESCRIPTIONS = {
        "Sleep stage W": "Sleep stage W",
        "Sleep stage 1": "Sleep stage 1",
        "Sleep stage 2": "Sleep stage 2",
        "Sleep stage 3": "Sleep stage 3",
        "Sleep stage 4": "Sleep stage 3",  # merge into N3
        "Sleep stage R": "Sleep stage R",
    }

    clean_onset, clean_duration, clean_description = [], [], []
    for onset, duration, description in zip(annot.onset, annot.duration, annot.description):
        desc = description.strip()
        if desc not in KEEP_DESCRIPTIONS:
            continue  # skip "?", "Movement time", and any other unknown label
        clean_onset.append(onset)
        clean_duration.append(duration)
        clean_description.append(KEEP_DESCRIPTIONS[desc])

    if not clean_description:
        raise ValueError(f"No usable sleep-stage annotations in {hyp_path.name}")

    annot = mne.Annotations(
        onset=clean_onset,
        duration=clean_duration,
        description=clean_description,
        orig_time=annot.orig_time,
    )
    raw.set_annotations(annot, emit_warning=False)

    # ---- Crop extra wake margins before/after sleep (common practice) ----
    if crop_wake_minutes is not None and len(annot) > 2:
        sleep_onset = annot[1]["onset"]
        sleep_offset = annot[-2]["onset"] + annot[-2]["duration"]
        tmin = max(0, sleep_onset - crop_wake_minutes * 60)
        tmax = min(raw.times[-1], sleep_offset + crop_wake_minutes * 60)
        raw.crop(tmin=tmin, tmax=tmax)

    # ---- Mapping now has ONLY unique values (no None, no duplicates) ----
    EVENT_ID = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage R": 4,
    }

    events, event_id_map = mne.events_from_annotations(
        raw, event_id=EVENT_ID, chunk_duration=epoch_sec
    )

    epochs = mne.Epochs(
        raw, events, event_id=event_id_map,
        tmin=0.0, tmax=epoch_sec - (1.0 / raw.info["sfreq"]),
        baseline=None, preload=True, verbose=False,
    )

    X = epochs.get_data()                 # (n_epochs, n_channels, n_samples)
    y = epochs.events[:, 2]               # numeric sleep stage code for each epoch
    onsets = epochs.events[:, 0] / raw.info["sfreq"]  # seconds from start of cropped raw

    return X.astype(np.float32), y.astype(np.int64), onsets.astype(np.float64), available


# ----------------------------------------------------------------------
# 3) اجرای کل خط لوله روی همه رکوردها
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ساخت دیتاست epoch-محور از Sleep-EDF Expanded")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="مسیر پوشه sleep-edf-database-expanded-1.0.0")
    parser.add_argument("--output_dir", type=str, default="processed",
                         help="مسیر خروجی برای ذخیره metadata.csv و signals/*.npz")
    parser.add_argument("--epoch_sec", type=int, default=30,
                         help="طول هر epoch به ثانیه (پیش‌فرض استاندارد: ۳۰)")
    parser.add_argument("--channels", type=str, nargs="+",
                         default=["EEG Fpz-Cz", "EEG Pz-Oz"],
                         help='کانال‌هایی که نگه داشته می‌شوند. '
                              'برای مدل تک‌کاناله مثلا: --channels "EEG Fpz-Cz"')
    parser.add_argument("--crop_wake_minutes", type=float, default=30.0,
                         help="حاشیه بیداری قبل/بعد از خواب که نگه داشته می‌شود (دقیقه). "
                              "برای غیرفعال کردن: -1")
    parser.add_argument("--limit", type=int, default=None,
                         help="فقط N رکورد اول را پردازش کن (برای تست سریع)")
    parser.add_argument("--resume", action="store_true",
                         help="رکوردهایی که npz دارند دوباره پردازش نشوند "
                              "(متادیتای قبلی از metadata.csv خوانده می‌شود)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    signals_dir = output_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    crop_minutes = None if args.crop_wake_minutes < 0 else args.crop_wake_minutes

    # ---- در حالت resume، متادیتای قبلی را می‌خوانیم ----
    existing_metadata = None
    metadata_path = output_dir / "metadata.csv"
    if args.resume and metadata_path.exists():
        existing_metadata = pd.read_csv(metadata_path)
        print(f"📥 متادیتای قبلی خوانده شد: {len(existing_metadata)} ردیف")
        done_records = set(existing_metadata["record_id"].unique())
        print(f"📥 رکوردهای قبلاً پردازش‌شده: {len(done_records)}")
    else:
        done_records = set()

    pairs = find_pairs(data_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"تعداد جفت‌های PSG+Hypnogram پیدا شده: {len(pairs)}")

    all_rows = []
    failed_rows = []
    n_ok, n_fail, n_skip = 0, 0, 0

    for i, (psg_path, hyp_path) in enumerate(pairs, start=1):
        try:
            info = parse_filename(psg_path)
            npz_path = signals_dir / f"{info['record_id']}.npz"

            # ---- در حالت resume، اگر رکورد قبلا پردازش شده، متادیتای قبلی را اضافه کن ----
            if args.resume and info["record_id"] in done_records:
                prev_rows = existing_metadata[
                    existing_metadata["record_id"] == info["record_id"]
                ]
                all_rows.extend(prev_rows.to_dict("records"))
                n_skip += 1
                print(f"[{i}/{len(pairs)}] ⏭️  رد شد (متادیتای قبلی بازگردانی شد): {info['record_id']}")
                continue

            X, y, onsets, used_channels = extract_epochs_for_record(
                psg_path, hyp_path,
                epoch_sec=args.epoch_sec,
                channels=args.channels,
                crop_wake_minutes=crop_minutes,
            )

            np.savez_compressed(npz_path, X=X, channels=np.array(used_channels))

            for epoch_idx in range(len(y)):
                all_rows.append({
                    "subject_id": info["subject_id"],
                    "night": info["night"],
                    "session": info["session"],
                    "record_id": info["record_id"],
                    "epoch_idx": epoch_idx,
                    "onset_sec": onsets[epoch_idx],
                    "stage_id": int(y[epoch_idx]),
                    "stage_name": ID_TO_STAGE_NAME.get(int(y[epoch_idx]), "UNKNOWN"),
                    "signal_file": str(npz_path.relative_to(output_dir)),
                })

            n_ok += 1
            print(f"[{i}/{len(pairs)}] ✅ {info['record_id']}  "
                  f"({len(y)} epoch, subject={info['subject_id']}, night={info['night']})")

        except Exception as e:
            n_fail += 1
            err_msg = str(e)
            print(f"[{i}/{len(pairs)}] ❌ {psg_path.name}  خطا: {err_msg}")
            failed_rows.append({
                "psg_file": psg_path.name,
                "hyp_file": hyp_path.name,
                "error_type": type(e).__name__,
                "error_message": err_msg,
                "traceback": traceback.format_exc(),
            })

    metadata = pd.DataFrame(all_rows)
    metadata.to_csv(metadata_path, index=False)

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_path = output_dir / "failed_records.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"\n📋 جزئیات کامل خطاها ذخیره شد در: {failed_path}")
        print("خلاصه انواع خطا:")
        print(failed_df["error_type"].value_counts())

    print("\n========== خلاصه ==========")
    print(f"رکوردهای موفق: {n_ok}  |  رکوردهای ناموفق: {n_fail}  |  رد شده (resume): {n_skip}")
    print(f"تعداد کل epoch ها: {len(metadata)}")
    print(f"تعداد افراد یکتا (subject_id): {metadata['subject_id'].nunique()}")
    print(f"توزیع مراحل خواب:\n{metadata['stage_name'].value_counts()}")
    print(f"\nمتادیتا ذخیره شد در: {metadata_path}")
    print(f"سیگنال‌ها ذخیره شدند در: {signals_dir}/")

    
if __name__ == "__main__":
    main()