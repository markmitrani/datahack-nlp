"""
Step 3: compute voice-quality metrics (jitter, shimmer, HNR) per vowel segment.

This module is pure logic — no FastAPI imports — so it can be called by
`server.py`, from the CLI, or from a notebook without pulling in web deps.

Metrics computed (Praat conventions via parselmouth):
  • jitter     — local jitter, reported as percent (fraction × 100).
  • shimmer    — local shimmer, reported as percent (fraction × 100).
  • hnr        — mean harmonicity (harmonic-to-noise ratio), decibels.

We intentionally use Praat's built-in averaging rather than re-implementing
the paper's per-cycle hierarchical clustering outlier removal. Praat already
computes these as a mean over detected glottal cycles within the segment,
which is the standard clinical definition. The paper's extra outlier filtering
is applied at the aggregate step (compute_aggregate.py) where the user can
tune thresholds.

CLI usage:
    python compute_metrics.py <vowels.csv> <audio_dir> <output.csv>

    vowels.csv  — output of isolate_vowels.py (columns: filename, phoneme,
                  start, end, duration).
    audio_dir   — folder with the audio/video files referenced by filename.
    output.csv  — where to write the metrics CSV.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

# parselmouth is imported lazily inside the compute functions so this module
# can be imported (e.g. for type checks) even without parselmouth installed.


REQUIRED_INPUT_COLS = ["filename", "phoneme", "start", "end", "duration"]
METRIC_COLS = ["jitter", "shimmer", "hnr"]
OUTPUT_COLS = REQUIRED_INPUT_COLS + METRIC_COLS

# Praat parameters — defaults consistent with the paper's F0 range (70-300 Hz)
# and standard clinical settings for jitter/shimmer extraction.
PITCH_FLOOR_HZ = 70.0
PITCH_CEILING_HZ = 300.0
PERIOD_FLOOR = 0.0001
PERIOD_CEILING = 0.02
MAX_PERIOD_FACTOR = 1.3
MAX_AMPLITUDE_FACTOR = 1.6


# ── Single-segment metric extraction ────────────────────────────────────────

def _metrics_for_segment(segment: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    """Return (jitter_pct, shimmer_pct, hnr_db) for one audio slice.

    Returns NaN for any metric Praat can't compute (too short, unvoiced, etc.).
    """
    import parselmouth  # lazy

    nan = float("nan")
    try:
        sound = parselmouth.Sound(
            values=segment.astype(np.float64),
            sampling_frequency=float(sample_rate),
        )
    except Exception:
        return nan, nan, nan

    # Pitch + PointProcess are shared by jitter and shimmer.
    try:
        pitch = sound.to_pitch(time_step=0.0, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
        point_process = parselmouth.praat.call(
            [sound, pitch], "To PointProcess (cc)"
        )
    except Exception:
        pitch = None
        point_process = None

    jitter_pct = nan
    shimmer_pct = nan
    if point_process is not None:
        try:
            j = parselmouth.praat.call(
                point_process, "Get jitter (local)",
                0, 0, PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR,
            )
            jitter_pct = _guard(j) * 100.0 if _guard(j) is not None else nan
        except Exception:
            pass
        try:
            s = parselmouth.praat.call(
                [sound, point_process], "Get shimmer (local)",
                0, 0, PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR, MAX_AMPLITUDE_FACTOR,
            )
            shimmer_pct = _guard(s) * 100.0 if _guard(s) is not None else nan
        except Exception:
            pass

    hnr_db = nan
    try:
        harmonicity = sound.to_harmonicity_cc(
            time_step=0.01, minimum_pitch=PITCH_FLOOR_HZ, silence_threshold=0.1, periods_per_window=4.5,
        )
        v = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        g = _guard(v)
        if g is not None:
            hnr_db = g
    except Exception:
        pass

    return jitter_pct, shimmer_pct, hnr_db


def _guard(v: Any) -> float | None:
    """Reject None, NaN, Infinity, or absurdly large Praat sentinel values."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv) or abs(fv) > 1e10:
        return None
    return fv


# ── Per-file API ────────────────────────────────────────────────────────────

def compute_metrics_for_audio(
    audio_array: np.ndarray,
    sample_rate: int,
    phonemes_df: pd.DataFrame,
    filename: str | None = None,
) -> pd.DataFrame:
    """Compute metrics for every row in `phonemes_df`.

    Parameters
    ----------
    audio_array : 1-D float array of mono audio samples.
    sample_rate : samples/sec.
    phonemes_df : must have columns `phoneme`, `start`, `end` (seconds), plus
                  `duration` (ms). A `filename` column is optional; if missing
                  and `filename` is given, it's added.
    filename    : value to use for the filename column if not already in df.

    Returns a DataFrame with columns
        filename, phoneme, start, end, duration, jitter, shimmer, hnr.
    Jitter/shimmer are in percent; HNR is in dB. NaN for segments Praat can't
    analyze (too short, unvoiced, etc.).
    """
    df = phonemes_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("phoneme", "start", "end"):
        if col not in df.columns:
            raise ValueError(f"phonemes_df missing required column: {col}")
    if "duration" not in df.columns:
        df["duration"] = (df["end"].astype(float) - df["start"].astype(float)) * 1000.0
    if "filename" not in df.columns:
        df["filename"] = filename or ""

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = np.asarray(audio_array, dtype=np.float32)
    n_samples = len(audio_array)
    file_dur = n_samples / sample_rate

    jitters, shimmers, hnrs = [], [], []
    for _, row in df.iterrows():
        start_s = float(row["start"])
        end_s = min(float(row["end"]), file_dur)
        i0 = max(0, int(round(start_s * sample_rate)))
        i1 = min(n_samples, int(round(end_s * sample_rate)))
        if i1 <= i0:
            jitters.append(float("nan"))
            shimmers.append(float("nan"))
            hnrs.append(float("nan"))
            continue
        seg = audio_array[i0:i1]
        j, s, h = _metrics_for_segment(seg, sample_rate)
        jitters.append(j)
        shimmers.append(s)
        hnrs.append(h)

    df["jitter"] = jitters
    df["shimmer"] = shimmers
    df["hnr"] = hnrs
    # Round to something sane.
    for c in ("jitter", "shimmer"):
        df[c] = df[c].astype(float).round(4)
    df["hnr"] = df["hnr"].astype(float).round(3)
    return df[OUTPUT_COLS]


# ── Batch API (iterate a vowels CSV + audio folder) ─────────────────────────

SR_TARGET = 16000
AUDIO_EXTENSIONS = {".wav"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4a", ".mkv", ".webm"}


def _load_audio_any(path: Path) -> tuple[np.ndarray, int]:
    """Load audio from wav directly, or extract 16 kHz mono via ffmpeg for video."""
    if path.suffix.lower() in AUDIO_EXTENSIONS:
        arr, sr = sf.read(str(path), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr, int(sr)

    tmpdir = Path(tempfile.mkdtemp(prefix="metrics_"))
    try:
        wav_path = tmpdir / (path.stem + ".wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-vn", "-ac", "1", "-ar", str(SR_TARGET), "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        arr, sr = sf.read(str(wav_path), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr, int(sr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compute_metrics_batch(
    vowels_csv: str | Path,
    audio_dir: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load vowels_csv, find each unique filename inside audio_dir, compute metrics."""
    vowels_csv = Path(vowels_csv)
    audio_dir = Path(audio_dir)
    df = pd.read_csv(vowels_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ("filename", "phoneme", "start", "end") if c not in df.columns]
    if missing:
        raise ValueError(f"vowels CSV missing columns: {missing}")

    all_rows = []
    for fname, group in df.groupby("filename", sort=False):
        path = audio_dir / fname
        if not path.exists():
            # allow either the exact filename or matching stem
            candidates = list(audio_dir.glob(f"{Path(fname).stem}.*"))
            if not candidates:
                print(f"  [skip] audio not found for {fname}")
                continue
            path = candidates[0]
        print(f"  {fname}: loading audio ({path.name}) …", flush=True)
        t0 = time.time()
        try:
            audio, sr = _load_audio_any(path)
        except Exception as e:
            print(f"    failed to load audio: {e}")
            continue
        print(f"    {len(audio)/sr:.1f}s audio, running Praat on {len(group)} segments …", flush=True)
        out = compute_metrics_for_audio(audio, sr, group, filename=fname)
        all_rows.append(out)
        print(f"    done in {time.time()-t0:.1f}s", flush=True)

    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=OUTPUT_COLS)
    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"  wrote {len(result)} rows → {out_path}")
    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def _main(argv: list[str]) -> None:
    if len(argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    vowels_csv = argv[1]
    audio_dir = argv[2]
    output_csv = argv[3] if len(argv) == 4 else "output/metrics.csv"
    print(f"Reading vowels CSV: {vowels_csv}")
    print(f"Audio dir:          {audio_dir}")
    print(f"Output CSV:         {output_csv}")
    t0 = time.time()
    compute_metrics_batch(vowels_csv, audio_dir, output_csv)
    print(f"Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    _main(sys.argv)
