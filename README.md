# audiotwin

Lightweight, open-source (MIT) detection of **audio twins** — two files that
are the *same master recording*, regardless of encoding, bitrate, or source.

`audiotwin` takes two audio files and returns **raw similarity scores**. It is
deliberately unopinionated: it makes no business decisions (dedup merging,
canonical-track selection, thresholds tuned for your catalog). Those stay with
the caller. Think of it as a fast pre-filter for duplicate detection in a
larger pipeline.

## Installation

```bash
pip install audiotwin
```

`audiotwin` relies on the **`fpcalc`** command-line tool (from Chromaprint), a
system dependency you must install separately:

```bash
# Debian / Ubuntu
sudo apt-get install libchromaprint-tools

# macOS (Homebrew)
brew install chromaprint

# Windows
winget install --id AcoustID.Chromaprint -e
```

`fpcalc` bundles its own audio decoder, so no separate `ffmpeg` install or
Python bindings to `libchromaprint` are required — `audiotwin` only shells out
to the `fpcalc` binary, which keeps things portable (Chromaprint no longer
ships a Windows `.dll`, only the standalone executable). If `fpcalc` isn't on
`PATH`, point `audiotwin` at it via the `AUDIOTWIN_FPCALC` environment
variable.

## Quick start

```python
from audiotwin import detect

result = detect("track_a.mp3", "track_b.flac")
print(result["is_duplicate"], result["confidence"])
```

`detect` returns:

```python
{
    "track_a": "track_a.mp3",
    "track_b": "track_b.flac",
    "file_hash_match": False,
    "chromaprint_score": 0.97,       # 0.0–1.0
    "nfp_score": None,
    "nfp_segments_matched": None,
    "nfp_coverage": None,
    "is_duplicate": True,
    "confidence": 0.873,             # 0.0–1.0
}
```

### CLI

```bash
audiotwin compare track_a.mp3 track_b.flac        # human-readable
audiotwin compare track_a.mp3 track_b.flac --json # machine-readable
audiotwin fingerprint track.mp3                   # just the fingerprint
```

## The pipeline — three levels

`audiotwin` runs cheapest-signal-first and short-circuits as soon as it can.

**Level 0 — file hash.** `file_hash(path)` returns the SHA256 of the raw
bytes. Two identical hashes are byte-for-byte identical files — an instant,
certain duplicate. When this matches, nothing else is computed.

**Level 1 — Chromaprint.** `compute_fingerprint(path)` shells out to `fpcalc`
to decode the first `max_duration` seconds (default 120) to mono/44100 Hz and
compute a Chromaprint acoustic fingerprint. `compare_fingerprints(a, b)` does
the standard AcoustID bit-wise comparison and returns a `0.0–1.0` similarity.
This survives re-encoding, bitrate changes, and format conversion.

**Level 2 — NFP (optional, caller-supplied).** To keep the dependency
footprint tiny, `audiotwin` does **not** compute neural embeddings itself.
Instead, `combine_scores(...)` accepts a neural-fingerprint (NFP) similarity
you computed elsewhere and folds it into the verdict as a second confirmation.

### Decision logic

```
file_hash_match ─── yes ──▶ duplicate,  confidence = 1.0   (stop)
        │
        no
        ▼
chromaprint_score ≥ 0.85 ? ── no ──▶ not a duplicate, confidence = 0.0
        │
       yes
        ▼
   nfp_score provided?
        ├── no ─────────────▶ duplicate, confidence = chromaprint × 0.9
        ├── yes, ≥ 0.90 ────▶ duplicate, confidence = (chromaprint + nfp) / 2
        └── yes, < 0.90 ────▶ NOT a duplicate, confidence = 0.0
                              (NFP contradicts Chromaprint → reject)
```

That last branch is the system's **anti-false-positive guard**: when a neural
fingerprint disagrees with a strong Chromaprint match, `audiotwin` rejects the
pair rather than risk a false positive.

The thresholds (`0.85`, `0.90`) are the defaults; both are configurable via
`chromaprint_threshold` / `nfp_threshold` parameters on `detect` and
`combine_scores`.

## Limitations

Be honest about what this does and does not do:

- **Same master only.** `audiotwin` detects the *same recording* re-encoded or
  re-sourced. It does **not** detect covers, live versions, remixes, or remasters
  — those are different audio and will score low by design.
- **≥ 10 seconds of audio required.** Shorter clips don't yield a reliable
  fingerprint; `compute_fingerprint` raises `AudioTooShortError`.
- **NFP is optional but improves precision.** Chromaprint alone is strong but
  can be fooled by very similar-sounding but distinct audio. Rough precision:
  **~98% with Chromaprint alone vs. ~99.5% when an NFP score is supplied.** If
  you have a neural fingerprinter available, feed its score in.

## Scope

`audiotwin` returns scores — nothing more. It does **not** do source
separation, transcription, classification, canonical-track selection, or
duplicate merging. Those decisions belong to the calling application.

## License

MIT — see [LICENSE](LICENSE). Copyright to be set by the user.

---

<sub>Built to power duplicate detection at Mkzik.</sub>
