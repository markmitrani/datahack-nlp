import whisperx
import json
import time

AUDIO_FILE = "data/Aardema_maiden_t.wav"  # extracted audio
MODEL_SIZE = "large-v3"
LANGUAGE = "nl"  # Dutch
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # CPU-friendly quantization

print(f"Loading Whisper model ({MODEL_SIZE})...")
t0 = time.time()
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)
print(f"  loaded in {time.time()-t0:.1f}s")

print("Loading audio...")
audio = whisperx.load_audio(AUDIO_FILE)

print("Transcribing...")
t0 = time.time()
result = model.transcribe(audio, batch_size=4, language=LANGUAGE)
print(f"  transcribed in {time.time()-t0:.1f}s")

print("Loading alignment model (Dutch)...")
t0 = time.time()
model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
print(f"  loaded in {time.time()-t0:.1f}s")

print("Aligning...")
t0 = time.time()
result_aligned = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    DEVICE,
    return_char_alignments=True,  # character-level, finer than word
)
print(f"  aligned in {time.time()-t0:.1f}s")

# Save full output for inspection
with open("whisperx_output.json", "w") as f:
    json.dump(result_aligned, f, indent=2, ensure_ascii=False)

print("Saved whisperx_output.json")