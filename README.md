# Voice Parameter Extraction from Continuous Speech

Extract jitter, shimmer, and harmonicity from audio files of continuous speech by isolating sustained vowel segments.

## Objective

Jitter, shimmer, and harmonicity are voice quality measures conventionally computed on sustained vowel recordings, not continuous speech. This repository aims to bridge that gap: given an audio file of continuous speech, automatically detect segments where a speaker sustains a vowel sound (e.g., "uh", "ee", "aa") for at least ~80 ms, extract the voice parameters from those segments only, and aggregate them per participant.

This approach is based on the procedure described in:

> Nathan, V., Rahman, M. M., Vatanparvar, K., Nemati, E., Blackstock, E., & Kuang, J. (2019, November). Extraction of voice parameters from continuous running speech for pulmonary disease monitoring. In *2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)* (pp. 859–864). IEEE. [link]

## Pipeline

The script performs the following steps:

1. **Process an audio file and identify speech** — load the audio and apply voice activity detection to isolate speech regions from silence/noise.
2. **Identify sustained vowel segments** — within the speech regions, detect parts where the participant sustains a vowel sound (e.g., holds "aa" for at least 80 ms). This requires segmenting by phoneme and filtering for vowels that persist above the minimum duration threshold. *This step is expected to be the bulk of the work.*
3. **Extract voice parameters** — compute jitter, shimmer, and harmonicity (HNR) from each identified sustained-vowel segment using an open-source voice analysis tool (e.g., Praat via Parselmouth, or openSMILE).
4. **Average across the file** — aggregate the per-segment values into a single set of values per participant (per audio file).
5. **Store output as CSV** — write one row per participant with the averaged jitter, shimmer, and harmonicity values.

## Input / Output

- **Input:** audio file(s) of continuous speech (one per participant).
- **Output:** a CSV file with columns for participant ID, mean jitter, mean shimmer, and mean harmonicity.

## Configurable parameters

- Minimum sustained-vowel duration (default: 80 ms)
- Voice analysis backend (Praat / openSMILE / other)
- VAD sensitivity
- Aggregation method (mean, median, etc.)

## Status

Work in progress. See referred paper for ideas on implementation directions.