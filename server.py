from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Phoneme Metrics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.post("/compute_metrics")
async def compute_metrics(
    audio_file: UploadFile = File(...),
    csv_content: str = Form(...),
    csv_name: str = Form(...),
) -> list[dict[str, Any]]:
    # TODO remove everything below and substitute the actual logic from separate script(s).
    # audio_bytes = await audio_file.read()
    # try:
    #     audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=422,
    #         detail=f"Cannot decode audio (WAV required): {e}. Convert with: ffmpeg -i input.mp4 -vn -ar 16000 out.wav",
    #     )

    # if audio_array.ndim > 1:
    #     audio_array = audio_array.mean(axis=1)

    # total_samples = len(audio_array)
    # file_duration = total_samples / sample_rate

    # try:
    #     df_in = pd.read_csv(io.StringIO(csv_content))
    #     df_in.columns = [c.strip().lower() for c in df_in.columns]
    #     for col in ("phoneme", "start", "end"):
    #         if col not in df_in.columns:
    #             raise ValueError(f"Missing column: {col}")
    # except Exception as e:
    #     raise HTTPException(status_code=422, detail=f"Invalid CSV: {e}")

    # results: list[dict[str, Any]] = []
    # filename = audio_file.filename or "audio"

    # for _, row in df_in.iterrows():
    #     phoneme = str(row["phoneme"])
    #     start_s = float(row["start"])
    #     end_s = min(float(row["end"]), file_duration)
    #     duration_ms = (end_s - start_s) * 1000.0

    #     i_start = int(round(start_s * sample_rate))
    #     i_end = int(round(end_s * sample_rate))

    #     if i_end <= i_start or i_start >= total_samples:
    #         results.append(_nan_row(filename, phoneme, start_s, end_s, duration_ms))
    #         continue

    #     segment = audio_array[i_start:i_end]
    #     jitter, shimmer, hnr = _compute_metrics(segment, sample_rate)

    #     results.append({
    #         "filename": filename,
    #         "phoneme": phoneme,
    #         "start": round(start_s, 4),
    #         "end": round(end_s, 4),
    #         "duration": round(duration_ms, 3),
    #         "jitter": _safe_json(jitter),
    #         "shimmer": _safe_json(shimmer),
    #         "hnr": _safe_json(hnr),
    #     })

    # out_path = OUTPUT_DIR / f"{csv_name}_metrics.csv"
    # pd.DataFrame(results).to_csv(out_path, index=False)
    return None


def _compute_metrics(
    segment: np.ndarray,
    sample_rate: int,
) -> tuple[float, float, float]:
    # TODO this helper is probably not even needed.
    # try:
    #     sound = parselmouth.Sound(
    #         values=segment.astype(np.float64),
    #         sampling_frequency=float(sample_rate),
    #     )
    #     pitch = sound.to_pitch()

    #     jitter_raw = parselmouth.praat.call(
    #         [sound, pitch], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    #     )
    #     jitter = _guard(jitter_raw) * 100.0 if _guard(jitter_raw) is not None else float("nan")

    #     shimmer = _guard(
    #         parselmouth.praat.call(
    #             [sound, pitch], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    #         )
    #     )
    #     if shimmer is None:
    #         shimmer = float("nan")

    #     harmonicity = sound.to_harmonicity()
    #     hnr = _guard(parselmouth.praat.call(harmonicity, "Get mean", 0, 0))
    #     if hnr is None:
    #         hnr = float("nan")

    #     return jitter, shimmer, hnr
    # except Exception:
    #     return float("nan"), float("nan"), float("nan")
    return None, None, None


def _guard(v: Any) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or abs(fv) > 1e10:
        return None
    return fv


def _safe_json(v: float) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v, 6)


def _nan_row(
    filename: str, phoneme: str, start: float, end: float, duration_ms: float
) -> dict[str, Any]:
    return {
        "filename": filename,
        "phoneme": phoneme,
        "start": round(start, 4),
        "end": round(end, 4),
        "duration": round(duration_ms, 3),
        "jitter": None,
        "shimmer": None,
        "hnr": None,
    }


class AggregateRequest(BaseModel):
    csv_name: str


@app.post("/compute_aggregate")
def compute_aggregate(req: AggregateRequest) -> list[dict[str, Any]]:
    # TODO remove everything below and substitute the actual logic from separate script(s).
    # metrics_path = OUTPUT_DIR / f"{req.csv_name}_metrics.csv"
    # if not metrics_path.exists():
    #     raise HTTPException(
    #         status_code=404,
    #         detail=f"{metrics_path} not found. Run /compute_metrics first.",
    #     )

    # df = pd.read_csv(metrics_path)
    # required = {"filename", "jitter", "shimmer", "hnr"}
    # missing = required - set(df.columns)
    # if missing:
    #     raise HTTPException(
    #         status_code=422, detail=f"Metrics CSV missing columns: {missing}"
    #     )

    # agg = (
    #     df.groupby("filename", dropna=False)
    #     .agg(
    #         n_vowels=("jitter", lambda x: x.notna().sum()),
    #         mean_jitter_local_pct=("jitter", "mean"),
    #         mean_shimmer_local_db=("shimmer", "mean"),
    #         mean_hnr_db=("hnr", "mean"),
    #     )
    #     .reset_index()
    # )

    # out_path = OUTPUT_DIR / f"{req.csv_name}_metrics_agg.csv"
    # agg.to_csv(out_path, index=False)

    # return agg.where(pd.notna(agg), other=None).to_dict(orient="records")
    return []


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/phoneme_player.html")


# Serve static files last so API routes take priority
app.mount("/", StaticFiles(directory=".", html=True), name="static")
