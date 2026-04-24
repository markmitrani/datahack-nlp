"""
Step 4: filter outlier segments and aggregate per-participant voice-quality stats.

Input : per-vowel metrics CSV produced by compute_metrics.py
        (columns: filename, phoneme, start, end, duration, jitter, shimmer, hnr).
Output: one row per filename with summary statistics of jitter, shimmer, hnr
        plus n_vowels and the most frequent vowel phoneme.

Outlier removal is done per-segment using four user-tunable thresholds:

    min_duration_ms  — drop segments shorter than this (paper: 80 ms).
    max_jitter_pct   — drop segments whose local jitter exceeds this (paper: 16%).
    max_shimmer      — drop segments whose local shimmer exceeds this (paper: 30%).
    min_hnr_db       — drop segments whose HNR is below this (typical: 0 dB
                       or higher if you only want clearly-voiced material).

Segments with NaN in any metric are always dropped (Praat couldn't analyze them).

For each remaining segment group per filename, we compute:
    n_vowels              — kept segment count
    most_frequent_vowel   — mode of `phoneme`
    jitter_mean/median/std/p25/p75
    shimmer_mean/median/std/p25/p75
    hnr_mean/median/std/p25/p75

CLI usage:
    python compute_aggregate.py <metrics.csv> <output.csv> \
        [min_duration_ms] [max_jitter_pct] [max_shimmer] [min_hnr_db]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRIC_COLS = ("jitter", "shimmer", "hnr")
STAT_NAMES = ("mean", "median", "std", "p25", "p75")

# Default thresholds match the paper's reported sanity-check ranges.
DEFAULT_MIN_DURATION_MS = 80.0
DEFAULT_MAX_JITTER_PCT = 16.0
DEFAULT_MAX_SHIMMER = 30.0
DEFAULT_MIN_HNR_DB = 0.0


# ── Filtering ───────────────────────────────────────────────────────────────

def apply_thresholds(
    metrics_df: pd.DataFrame,
    min_duration_ms: float = DEFAULT_MIN_DURATION_MS,
    max_jitter_pct: float = DEFAULT_MAX_JITTER_PCT,
    max_shimmer: float = DEFAULT_MAX_SHIMMER,
    min_hnr_db: float = DEFAULT_MIN_HNR_DB,
) -> pd.DataFrame:
    """Return the subset of `metrics_df` that survives the four thresholds.

    Rows with NaN in any metric column are always dropped; the threshold
    checks are applied only to finite values.
    """
    df = metrics_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"filename", "phoneme", "duration", *METRIC_COLS}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"metrics CSV missing columns: {sorted(missing)}")

    # Coerce to numeric so downstream comparisons never silently pass strings.
    for c in ("duration", *METRIC_COLS):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask = (
        df["duration"].ge(min_duration_ms)
        & df["jitter"].le(max_jitter_pct)
        & df["shimmer"].le(max_shimmer)
        & df["hnr"].ge(min_hnr_db)
        & df[list(METRIC_COLS)].notna().all(axis=1)
    )
    return df.loc[mask].reset_index(drop=True)


# ── Aggregation ─────────────────────────────────────────────────────────────

def _summary_for_series(values: pd.Series) -> dict[str, float]:
    """mean / median / std / p25 / p75 with NaN guard."""
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return {k: float("nan") for k in STAT_NAMES}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        # ddof=1 sample std matches pandas default. If only 1 sample, std is NaN.
        "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
    }


def _most_frequent(values: Iterable) -> str:
    s = pd.Series(list(values)).dropna()
    if s.empty:
        return ""
    mode = s.mode()
    return str(mode.iloc[0]) if not mode.empty else ""


def aggregate(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Compute one summary row per filename from an already-filtered metrics df."""
    rows: list[dict] = []
    for fname, group in filtered_df.groupby("filename", sort=False):
        row: dict = {
            "filename": fname,
            "n_vowels": int(len(group)),
            "most_frequent_vowel": _most_frequent(group["phoneme"]),
        }
        for metric in METRIC_COLS:
            stats = _summary_for_series(group[metric])
            for stat_name, val in stats.items():
                row[f"{metric}_{stat_name}"] = val
        rows.append(row)

    cols = ["filename", "n_vowels", "most_frequent_vowel"] + [
        f"{m}_{s}" for m in METRIC_COLS for s in STAT_NAMES
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)

    # Round sensibly: stats in percent or dB fit comfortably in 3-4 decimals.
    for m in METRIC_COLS:
        for s in STAT_NAMES:
            col = f"{m}_{s}"
            out[col] = out[col].astype(float).round(4)
    return out


# ── Convenience API ─────────────────────────────────────────────────────────

def compute_aggregate(
    metrics_df: pd.DataFrame,
    min_duration_ms: float = DEFAULT_MIN_DURATION_MS,
    max_jitter_pct: float = DEFAULT_MAX_JITTER_PCT,
    max_shimmer: float = DEFAULT_MAX_SHIMMER,
    min_hnr_db: float = DEFAULT_MIN_HNR_DB,
) -> pd.DataFrame:
    """Convenience wrapper: apply thresholds and aggregate in one call."""
    filtered = apply_thresholds(
        metrics_df,
        min_duration_ms=min_duration_ms,
        max_jitter_pct=max_jitter_pct,
        max_shimmer=max_shimmer,
        min_hnr_db=min_hnr_db,
    )
    return aggregate(filtered)


def compute_aggregate_from_csv(
    metrics_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    min_duration_ms: float = DEFAULT_MIN_DURATION_MS,
    max_jitter_pct: float = DEFAULT_MAX_JITTER_PCT,
    max_shimmer: float = DEFAULT_MAX_SHIMMER,
    min_hnr_db: float = DEFAULT_MIN_HNR_DB,
) -> pd.DataFrame:
    metrics_df = pd.read_csv(metrics_csv)
    result = compute_aggregate(
        metrics_df,
        min_duration_ms=min_duration_ms,
        max_jitter_pct=max_jitter_pct,
        max_shimmer=max_shimmer,
        min_hnr_db=min_hnr_db,
    )
    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"  wrote {len(result)} rows → {out_path}")
    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def _main(argv: list[str]) -> None:
    if len(argv) < 3 or len(argv) > 7:
        print(__doc__)
        sys.exit(1)
    metrics_csv = argv[1]
    output_csv = argv[2]

    def _getf(idx: int, default: float) -> float:
        return float(argv[idx]) if len(argv) > idx else default

    min_dur = _getf(3, DEFAULT_MIN_DURATION_MS)
    max_j = _getf(4, DEFAULT_MAX_JITTER_PCT)
    max_s = _getf(5, DEFAULT_MAX_SHIMMER)
    min_h = _getf(6, DEFAULT_MIN_HNR_DB)

    print(f"Reading metrics CSV: {metrics_csv}")
    print(
        f"Thresholds: min_duration_ms={min_dur}  max_jitter_pct={max_j}  "
        f"max_shimmer={max_s}  min_hnr_db={min_h}"
    )
    df = compute_aggregate_from_csv(
        metrics_csv,
        output_csv,
        min_duration_ms=min_dur,
        max_jitter_pct=max_j,
        max_shimmer=max_s,
        min_hnr_db=min_h,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    _main(sys.argv)
