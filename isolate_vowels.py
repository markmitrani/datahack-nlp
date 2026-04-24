#!/usr/bin/env python3
"""
Steps 1 + 2 of the voice-parameter pipeline: phoneme segmentation, keeping vowels.

Given a Dutch speech recording (mp4 or wav), this script:
  1. Extracts 16 kHz mono audio (ffmpeg) if the input is mp4.
  2. Loads facebook/wav2vec2-lv-60-espeak-cv-ft — a multilingual phoneme
     recognizer that outputs eSpeak IPA. It handles Dutch without any
     target-language flag.
  3. Runs CTC inference in non-overlapping chunks on CPU to keep memory bounded.
  4. Takes argmax over the vocabulary → one predicted token per 20 ms frame.
  5. Turns per-frame predictions into phoneme segments with realistic durations
     (see collapse_to_segments doc for the CTC-blipping heuristic).
  6. Filters to IPA vowels (first char of the phoneme token) and writes CSV.

Important note on the tokenizer: the HF tokenizer for this model pulls in the
`phonemizer` library, which needs the `espeak` / `espeak-ng` binary. We don't
need the tokenizer at all — we only decode ids to phoneme strings via vocab.json.
So we load only the feature extractor + model, and read vocab.json directly.
That keeps the script dependency-light: torch, transformers, soundfile,
huggingface_hub, numpy, pandas — NO espeak required.

Usage:
    python isolate_vowels.py <input.mp4|wav> <output.csv> [model_dir]

If model_dir is omitted, the model is auto-downloaded from Hugging Face (cached).
"""
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

# wav2vec2 has total stride 320 at 16 kHz → one output frame per 20 ms → 50 Hz.
FRAME_HZ = 50.0
SR = 16000

# IPA vowel characters. A phoneme token is treated as a vowel if its first char
# is in this set — this correctly catches Dutch monophthongs and diphthongs
# (aː, eː, ɛi, ɔʏ, ʌu, aɪ, …) since eSpeak writes the vowel nucleus first.
IPA_VOWELS = set("aɐɑɒæɘɜɚɛɝɞeəiɨɪɯɤoɔœøɶuʉʊʌʏy")

# Max gap (in 20 ms frames) between same-phoneme emissions to still be
# considered one cluster. Sustained vowels emit `a pad a pad a …`; a real pause
# between two separate /a/ tokens would be much longer. 15 frames = 300 ms.
MAX_INTRA_CLUSTER_GAP = 15

# When splitting pad frames between neighboring clusters, cap the extension
# from each cluster edge to at most this many frames. Without the cap, a
# phoneme at the edge of a long silence would inherit hundreds of ms of
# non-speech padding, distorting duration and jitter/shimmer measurements.
# 5 frames = 100 ms is comfortably larger than typical vowel durations.
MAX_EXTENSION_FRAMES = 5


def extract_audio_if_needed(path: str) -> tuple[str, Path | None]:
    """If path is video/non-wav, extract 16 kHz mono WAV to a temp file.

    Returns (wav_path, tmpdir_to_cleanup_or_None).
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return str(p), None
    tmpdir = Path(tempfile.mkdtemp(prefix="isolate_vowels_"))
    wav_path = tmpdir / (p.stem + ".wav")
    print(f"      extracting audio with ffmpeg → {wav_path}")
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


def load_vocab(model_source: str):
    """Read vocab.json → (id_to_token dict, pad_id).

    model_source is either a local directory containing vocab.json or a HF repo id.
    """
    p = Path(model_source) / "vocab.json"
    if p.exists():
        with open(p) as f:
            vocab = json.load(f)
    else:
        # Fetch from HF hub (cached).
        from huggingface_hub import hf_hub_download

        vocab_path = hf_hub_download(repo_id=model_source, filename="vocab.json")
        with open(vocab_path) as f:
            vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
    pad_id = vocab["<pad>"]
    return id_to_token, pad_id


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


def main(input_path: str, csv_path: str, model_source: str) -> None:
    t_all = time.time()

    print(f"[1/4] Loading model from {model_source} …")
    t0 = time.time()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_source)
    model = Wav2Vec2ForCTC.from_pretrained(model_source).eval()
    id_to_token, pad_id = load_vocab(model_source)
    print(
        f"      vocab={len(id_to_token)}  pad_id={pad_id}  "
        f"loaded in {time.time()-t0:.1f}s"
    )

    print(f"[2/4] Preparing audio: {input_path}")
    wav_path, tmpdir = extract_audio_if_needed(input_path)
    try:
        audio = load_audio(wav_path)
        dur_s = len(audio) / SR
        print(f"      {dur_s:.2f} s, {len(audio)} samples @ {SR} Hz mono")

        print("[3/4] Running CTC inference (CPU, 20 s chunks) …")
        t0 = time.time()
        pred_ids = chunked_predict(model, fe, audio)
        print(
            f"      {len(pred_ids)} frames "
            f"(expected ≈ {int(dur_s * FRAME_HZ)})  in {time.time()-t0:.1f}s"
        )

        print("[4/4] Collapsing to segments and filtering to vowels …")
        segments = collapse_to_segments(pred_ids, id_to_token, pad_id)
        rows = [
            {
                "phoneme": tok,
                "start": round(s / FRAME_HZ, 3),
                "end": round(e / FRAME_HZ, 3),
            }
            for tok, s, e in segments
            if is_vowel(tok)
        ]
        df = pd.DataFrame(rows, columns=["phoneme", "start", "end"])
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\nTotal phoneme segments: {len(segments)}")
    print(f"Vowel segments saved:   {len(df)}  →  {csv_path}")
    if len(df):
        print("\nFirst 15 rows:")
        print(df.head(15).to_string(index=False))
        print("\nVowel counts (top 10):")
        print(df["phoneme"].value_counts().head(10).to_string())
        durs = df["end"] - df["start"]
        print(
            f"\nSegment durations (s): mean={durs.mean():.3f}  "
            f"median={durs.median():.3f}  max={durs.max():.3f}  "
            f">=80 ms: {(durs >= 0.08).sum()} / {len(df)}"
        )
    print(f"\nTotal wall time: {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    input_path = sys.argv[1]
    csv_path = sys.argv[2]
    model_source = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_MODEL_ID
    main(input_path, csv_path, model_source)
