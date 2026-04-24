import json
import pandas as pd

INPUT = "whisperx_output.json"
OUTPUT = "vowels.csv"

# Dutch vowel letters (lowercase; we'll lowercase input)
VOWEL_CHARS = set("aeiouy")

with open(INPUT) as f:
    data = json.load(f)

# Flatten all char entries across all segments, keeping only chars with timestamps
chars = []
for seg in data["segments"]:
    for c in seg.get("chars", []):
        if "start" not in c or "end" not in c:
            continue  # skip spaces/punctuation without timing
        chars.append({
            "char": c["char"].lower(),
            "start": c["start"],
            "end": c["end"],
        })

# Merge adjacent vowel chars into vowel-group segments
# e.g. "aa" in "gaat" -> one segment spanning both 'a' chars
vowel_segments = []
i = 0
while i < len(chars):
    if chars[i]["char"] in VOWEL_CHARS:
        j = i
        # extend while next char is a vowel AND temporally contiguous
        while (
            j + 1 < len(chars)
            and chars[j + 1]["char"] in VOWEL_CHARS
            and abs(chars[j + 1]["start"] - chars[j]["end"]) < 0.02  # 20ms gap tolerance
        ):
            j += 1
        phoneme = "".join(chars[k]["char"] for k in range(i, j + 1))
        vowel_segments.append({
            "phoneme": phoneme,
            "start": chars[i]["start"],
            "end": chars[j]["end"],
            "duration": chars[j]["end"] - chars[i]["start"],
        })
        i = j + 1
    else:
        i += 1

df = pd.DataFrame(vowel_segments)
df.to_csv(OUTPUT, index=False)

print(f"Found {len(df)} vowel segments")
print(f"Duration range: {df['duration'].min()*1000:.0f}ms – {df['duration'].max()*1000:.0f}ms")
print(f"Segments >= 80ms: {(df['duration'] >= 0.080).sum()}")
print(f"\nFirst 15 rows:")
print(df.head(15).to_string(index=False))