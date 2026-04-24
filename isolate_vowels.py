#!/usr/bin/env python3
"""
Steps 1 + 2 of the voice-parameter pipeline: phoneme segmentation, keeping vowels.

Batch mode. Scans an input directory for speech recordings (mp4 / wav), runs a
phoneme recognizer on each, filters to IPA vowels, and writes ONE combined CSV
with columns:

    filename, phoneme, start, end, duration

(`start` and `end` are in seconds, `duration` is in milliseconds.)

What this does per file:
  1. Extracts 16 kHz mono audio (ffmpeg) if the input is mp4/video.
  2. Loads facebook/wav2vec2-lv-60-espeak-cv-ft — a multilingual phoneme
     recognizer that outputs eSpeak IPA. Dutch works without a target-language
     flag; the model is loaded once and reused across files.
  3. Runs CTC inference in non-overlapping chunks on CPU to bound memory.
  4. Takes argmax over the vocabulary → one predicted token per 20 ms frame.
  5. Turns per-frame predictions into phoneme segments with realistic durations
     (see collapse_to_segments for the CTC-blipping heuristic).
  6. Keeps only IPA vowels (first char of the phoneme token) and appends rows
     to the combined output.

Tokenizer sidestep: the HF tokenizer for this model pulls in `phonemizer`,
which needs espeak. We don't need the tokenizer at all — we decode ids to
phoneme strings via vocab.json. So we load only the feature extractor + model
and read vocab.json directly. Dependencies: torch, transformers, soundfile,
huggingface_hub, numpy, pandas. ffmpeg must be on PATH for mp4 input.

Usage:
    python isolate_vowels.py                                  # videos/ → output/vowels.csv
    python isolate_vowels.py <input_dir>                      # custom input dir
    python isolate_vowels.py <input_dir> <output.csv>         # custom output
    python isolate_vowels.py <input_dir> <output.csv> <model> # local dir or HF id
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

DEFAULT_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_INPUT_DIR = "videos"
DEFAULT_OUTPUT_CSV = "output/vowels.csv"

# wav2vec2 has total stride 320 at 16 kHz → one output frame per 20 ms → 50 Hz.
FRAME_HZ = 50.0
SR = 16000

# IPA vowel characters. A phoneme token is treated as a vowel if its first char
# is in this set — this correctly catches Dutch monophthongs and diphthongs
# (aː, eː, ɛi, ɔʏ, ʌu, aɪ, …) since eSpeak writes the vowel nucleus first.
IPA_VOWELS = set("aɐɑɒæɘɜɚɛɝɞeəiɨɪɯɤoɔœøɶuʉʊʌʏy")

# CTC decoding heuristics (see collapse_to_segments for rationale):
MAX_INTRA_CLUSTER_GAP = 15   # 300 ms — merges sustained vowels
MAX_EXTENSION_FRAMES = 5     # ±100 ms — caps extension into long silences

# Input extensions we'll pick up during a folder scan.
AUDIO_EXTENSIONS = {".wav"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4a", ".mkv", ".webm"}


# ── Audio loading ───────────────────────────────────────────────────────────

def extract_audio_if_needed(path: str) -> tuple[str, Path | None]:
    """If path is video/non-wav, extract 16 kHz mono WAV to a temp file.

    Returns (wav_path, tmpdir_to_cleanup_or_None).
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return str(p), None
    tmpdir = Path(tempfile.mkdtemp(prefix="isolate_vowels_"))
    wav_path = tmpdir / (p.stem + ".wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(p),
            "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return str(wav_path), tmpdir


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"expected {SR} Hz mono WAV, got {sr} Hz")
    return audio


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(model_source: str):
    """Returns (feature_extractor, model, id_to_token dict, pad_id)."""
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_source)
    model = Wav2Vec2ForCTC.from_pretrained(model_source).eval()
    id_to_token, pad_id = load_vocab(model_source)
    return fe, model, id_to_token, pad_id


def load_vocab(model_source: str):
    """Read vocab.json → (id_to_token dict, pad_id).

    model_source is either a local directory containing vocab.json or a HF repo id.
    """
    p = Path(model_source) / "vocab.json"
    if p.exists():
        with open(p) as f:
            vocab = json.load(f)
    else:
        from huggingface_hub import hf_hub_download
        vocab_path = hf_hub_download(repo_id=model_source, filename="vocab.json")
        with open(vocab_path) as f:
            vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
    pad_id = vocab["<pad>"]
    return id_to_token, pad_id


# ── Inference ───────────────────────────────────────────────────────────────

def chunked_predict(model, fe, audio, chunk_s=20.0) -> np.ndarray:
    """Run CTC inference in non-overlapping chunks; concat per-frame argmax ids."""
    chunk_samples = int(chunk_s * SR)
    preds = []
    with torch.inference_mode():
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) < 400:  # below conv receptive field
                break
            inputs = fe(chunk, sampling_rate=SR, return_tensors="pt")
            logits = model(inputs.input_values).logits  # (1, T, V)
            preds.append(logits.argmax(dim=-1).squeeze(0).cpu().numpy())
    return (
        np.concatenate(preds).astype(np.int64)
        if preds
        else np.array([], dtype=np.int64)
    )


def collapse_to_segments(pred_ids: np.ndarray, id_to_token: dict, pad_id: int):
    """Convert per-frame CTC argmax predictions into phoneme segments with
    realistic durations.

    Why this is non-trivial: wav2vec2-CTC emits each phoneme as a brief "blip"
    surrounded by pad frames (e.g. `a pad pad pad a pad pad t pad`). The argmax
    frame marks where the phoneme was most confidently detected — not its full
    duration. Using run-length alone would give every phoneme a 20 ms duration.

    Algorithm:
      1. Collect non-pad emissions: [(frame, token_id), …].
      2. Cluster consecutive same-phoneme emissions into (token, first, last)
         if they're within MAX_INTRA_CLUSTER_GAP of each other AND no other
         phoneme is emitted between them. This recovers sustained vowels.
      3. Assign each cluster start/end by splitting the pad gap to its
         neighbors at the midpoint, capped by MAX_EXTENSION_FRAMES per side
         so phonemes adjacent to long silences don't absorb all of it.

    Returns [(token_str, start_frame, end_frame), …] with fractional (float)
    frame boundaries that preserve sub-20 ms resolution.
    """
    n = len(pred_ids)
    if n == 0:
        return []

    # 1. Non-pad emissions.
    emissions = [(i, int(tok)) for i, tok in enumerate(pred_ids) if int(tok) != pad_id]
    if not emissions:
        return []

    # 2. Cluster consecutive same-phoneme emissions.
    clusters = []  # (token_id, first_frame, last_frame)
    tok0, frame0 = emissions[0][1], emissions[0][0]
    first, last = frame0, frame0
    for frame, t in emissions[1:]:
        if t == tok0 and (frame - last) <= MAX_INTRA_CLUSTER_GAP:
            last = frame
        else:
            clusters.append((tok0, first, last))
            tok0, first, last = t, frame, frame
    clusters.append((tok0, first, last))

    # 3. Midpoint split with capped extension.
    segments = []
    for i, (tok, first, last) in enumerate(clusters):
        prev_last = clusters[i - 1][2] if i > 0 else -1
        next_first = clusters[i + 1][1] if i < len(clusters) - 1 else n
        left_gap = first - prev_last
        right_gap = next_first - last
        left_ext = min(left_gap / 2.0, MAX_EXTENSION_FRAMES)
        right_ext = min(right_gap / 2.0, MAX_EXTENSION_FRAMES)
        seg_start = max(0.0, first - left_ext)
        seg_end = min(float(n), last + right_ext)
        segments.append((id_to_token[tok], seg_start, seg_end))
    return segments


def is_vowel(token: str) -> bool:
    if not token or token.startswith("<"):  # <pad>, <s>, </s>, <unk>
        return False
    return token[0] in IPA_VOWELS


# ── Per-file processing ─────────────────────────────────────────────────────

def process_file(
    path: str,
    fe,
    model,
    id_to_token: dict,
    pad_id: int,
) -> pd.DataFrame:
    """Run the phoneme pipeline on one file, return a vowels-only DataFrame.

    Columns: filename, phoneme, start, end, duration (duration in ms).
    filename is the original file name (with extension), e.g. "Aardema_maiden_t.mp4".
    """
    fname = Path(path).name
    wav_path, tmpdir = extract_audio_if_needed(path)
    try:
        audio = load_audio(wav_path)
        pred_ids = chunked_predict(model, fe, audio)
        segments = collapse_to_segments(pred_ids, id_to_token, pad_id)

        rows = []
        for tok, s, e in segments:
            if not is_vowel(tok):
                continue
            start_s = s / FRAME_HZ
            end_s = e / FRAME_HZ
            rows.append({
                "filename": fname,
                "phoneme": tok,
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "duration": round((end_s - start_s) * 1000, 3),
            })
        return pd.DataFrame(rows, columns=["filename", "phoneme", "start", "end", "duration"])
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def find_inputs(input_dir: str) -> list[Path]:
    """Return all audio/video files in input_dir (non-recursive), sorted."""
    d = Path(input_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    exts = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)


# ── Main ────────────────────────────────────────────────────────────────────

def main(input_dir: str, output_csv: str, model_source: str) -> None:
    t_all = time.time()

    inputs = find_inputs(input_dir)
    if not inputs:
        print(f"No audio/video files found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(inputs)} input file(s) in {input_dir}")
    for p in inputs:
        print(f"  · {p.name}")

    print(f"\nLoading model from {model_source} …")
    t0 = time.time()
    fe, model, id_to_token, pad_id = load_model(model_source)
    print(f"  model loaded in {time.time() - t0:.1f}s  (vocab={len(id_to_token)})")

    dfs = []
    for idx, path in enumerate(inputs, 1):
        print(f"\n[{idx}/{len(inputs)}] {path.name}")
        t0 = time.time()
        df = process_file(str(path), fe, model, id_to_token, pad_id)
        elapsed = time.time() - t0
        print(
            f"  {len(df)} vowel segments  "
            f"(mean duration {df['duration'].mean():.0f} ms, "
            f">=80 ms: {(df['duration'] >= 80).sum()})  "
            f"in {elapsed:.1f}s"
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["filename", "phoneme", "start", "end", "duration"]
    )
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    print(f"\n── Done ──")
    print(f"Files processed:  {len(inputs)}")
    print(f"Total segments:   {len(combined)}")
    print(f"Output CSV:       {out}")
    print(f"Total wall time:  {time.time() - t_all:.1f}s")
    if len(combined):
        print("\nPer-file summary:")
        summary = (
            combined.groupby("filename")
            .agg(n=("phoneme", "size"), mean_dur_ms=("duration", "mean"))
            .round(1)
        )
        print(summary.to_string())


if __name__ == "__main__":
    if len(sys.argv) > 4:
        print(__doc__)
        sys.exit(1)
    input_dir = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_INPUT_DIR
    output_csv = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_OUTPUT_CSV
    model_source = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_MODEL_ID
    main(input_dir, output_csv, model_source)
