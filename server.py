"""
FastAPI server for the phoneme-metrics UI.

All real work lives in two sibling modules:

    compute_metrics.py    — jitter / shimmer / HNR per vowel segment
    compute_aggregate.py  — threshold filtering + per-participant stats

This file only handles HTTP I/O, file decoding, and CSV persistence.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from compute_metrics import compute_metrics_for_audio
from compute_aggregate import (
    DEFAULT_MAX_JITTER_PCT,
    DEFAULT_MAX_SHIMMER,
    DEFAULT_MIN_DURATION_MS,
    DEFAULT_MIN_HNR_DB,
    compute_aggregate,
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Phoneme Metrics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── /compute_metrics ─────────────────────────────────────────────────────────

@app.post("/compute_metrics")
async def compute_metrics_endpoint(
    audio_file: UploadFile = File(...),
    csv_content: str = Form(...),
    csv_name: str = Form(...),
) -> list[dict[str, Any]]:
    """Run Praat metrics on a single uploaded WAV + the phonemes CSV for it.

    The CSV may contain rows for several filenames; we only process rows whose
    `filename` matches the uploaded audio (or all rows if the CSV has no
    `filename` column — legacy behaviour).
    """
    audio_bytes = await audio_file.read()
    try:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Cannot decode audio (WAV required): {e}. "
                "Convert with: ffmpeg -i input.mp4 -vn -ar 16000 out.wav"
            ),
        )

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = np.asarray(audio_array, dtype=np.float32)

    try:
        df_in = pd.read_csv(io.StringIO(csv_content))
        df_in.columns = [c.strip().lower() for c in df_in.columns]
        for col in ("phoneme", "start", "end"):
            if col not in df_in.columns:
                raise ValueError(f"Missing column: {col}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {e}")

    audio_name = audio_file.filename or "audio"
    # If the CSV has a filename column, restrict to rows that match this audio.
    if "filename" in df_in.columns:
        stem = Path(audio_name).stem
        mask = df_in["filename"].astype(str).apply(
            lambda f: f == audio_name or Path(str(f)).stem == stem
        )
        sub = df_in.loc[mask].copy()
        if sub.empty:
            # No matching rows — fall back to running on the whole CSV so the
            # user still gets output even if the filenames don't line up.
            sub = df_in.copy()
    else:
        sub = df_in.copy()

    try:
        out_df = compute_metrics_for_audio(
            audio_array, int(sample_rate), sub, filename=audio_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {e}")

    out_path = OUTPUT_DIR / f"{csv_name}_metrics.csv"
    # Merge with any existing metrics for the same csv_name: drop any rows
    # belonging to the same filename(s) we just re-computed, then append.
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
            existing = existing[~existing["filename"].isin(out_df["filename"].unique())]
            merged = pd.concat([existing, out_df], ignore_index=True)
        except Exception:
            merged = out_df
    else:
        merged = out_df
    merged.to_csv(out_path, index=False)

    # Convert NaN → None so FastAPI can emit valid JSON.
    records = out_df.to_dict(orient="records")
    for row in records:
        for k, v in list(row.items()):
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
    return records


# ── /compute_aggregate ───────────────────────────────────────────────────────

class AggregateRequest(BaseModel):
    csv_name: str
    min_duration_ms: float = DEFAULT_MIN_DURATION_MS
    max_jitter_pct: float = DEFAULT_MAX_JITTER_PCT
    max_shimmer: float = DEFAULT_MAX_SHIMMER
    min_hnr_db: float = DEFAULT_MIN_HNR_DB


@app.post("/compute_aggregate")
def compute_aggregate_endpoint(req: AggregateRequest) -> list[dict[str, Any]]:
    """Apply the 4 thresholds to a cached metrics CSV and return one row per file."""
    metrics_path = OUTPUT_DIR / f"{req.csv_name}_metrics.csv"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{metrics_path} not found. Run /compute_metrics first.",
        )

    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot read {metrics_path}: {e}")

    try:
        agg = compute_aggregate(
            df,
            min_duration_ms=req.min_duration_ms,
            max_jitter_pct=req.max_jitter_pct,
            max_shimmer=req.max_shimmer,
            min_hnr_db=req.min_hnr_db,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Aggregate failed: {e}")

    out_path = OUTPUT_DIR / f"{req.csv_name}_metrics_agg.csv"
    agg.to_csv(out_path, index=False)

    records = agg.to_dict(orient="records")
    for row in records:
        for k, v in list(row.items()):
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
    return records


# ── /filtered_segments ───────────────────────────────────────────────────────

@app.post("/filtered_segments")
def filtered_segments_endpoint(req: AggregateRequest) -> list[dict[str, Any]]:
    """Return the per-segment rows that survive the thresholds.

    Used by the UI to draw violin / strip plots of kept segments.
    """
    from compute_aggregate import apply_thresholds

    metrics_path = OUTPUT_DIR / f"{req.csv_name}_metrics.csv"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{metrics_path} not found. Run /compute_metrics first.",
        )
    df = pd.read_csv(metrics_path)
    try:
        filtered = apply_thresholds(
            df,
            min_duration_ms=req.min_duration_ms,
            max_jitter_pct=req.max_jitter_pct,
            max_shimmer=req.max_shimmer,
            min_hnr_db=req.min_hnr_db,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Filtering failed: {e}")
    records = filtered.to_dict(orient="records")
    for row in records:
        for k, v in list(row.items()):
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
    return records


# Serve static files last so API routes take priority
app.mount("/", StaticFiles(directory=".", html=True), name="static")
