# audiotwin

Lightweight, open-source (MIT) toolkit for detecting **relations between
audio files**: same master, remaster, edit, sample, mashup, cover,
instrumental.

`audiotwin` takes audio files (or precomputed features) and returns **raw
analysis data** — scores, hints, timestamps. It is deliberately
unopinionated: it makes no business decisions (dedup merging,
canonical-track selection, final verdicts tuned for your catalog). Those
stay with the caller. Think of it as the analysis layer of a larger
pipeline.

## Modules × relations

| Relation | What it means | Module / function | Extra needed |
|---|---|---|---|
| **DUPLICATE** | Same master, any encoding | `detect()` | core |
| **REMASTER** | Same recording, reworked signal (EQ/dynamics) | `classify_relation()` / `detect_relation()` | core |
| **EDIT** | Same recording, altered timeline (trim/extend/speed) | `classify_edit()` on caller-supplied match points | core |
| **SAMPLE** | Fragment reused inside another track | `landmark.LandmarkIndex` + `classify_sample()` | `[landmark]` |
| **MASHUP** | Fragments from several tracks combined | `landmark` + `classify_mashup()` | `[landmark]` |
| **COVER** | Same composition, new performance | `cover.cover_similarity()` | `[cover]` |
| **INSTRUMENTAL** | Same content, vocals present vs absent | `classify_instrumental_pair()` | core |
| *(aggregation)* | Ordered hypotheses from any of the above | `suggest_relation()` | core |

## Installation

```bash
pip install audiotwin              # core: DUPLICATE/REMASTER/EDIT/INSTRUMENTAL
pip install "audiotwin[landmark]"  # + SAMPLE/MASHUP (adds scipy)
pip install "audiotwin[cover]"     # + COVER (adds librosa)
pip install "audiotwin[all]"       # all analysis extras (NOT torch)

# Straight from git (note the 'name[extra] @ URL' syntax — the old
# '#egg=name[extra]' fragment is rejected by modern pip):
pip install "audiotwin[all] @ git+https://github.com/ZeWills987/audiotwin.git"

# Optional neural signal (PyTorch ~2 GB — deliberately outside [all]):
pip install "audiotwin[neural]"
pip install -e "git+https://github.com/sony/sampleid.git#egg=sampleid"
```

System dependencies:

```bash
# Debian / Ubuntu
sudo apt-get install libchromaprint-tools ffmpeg

# macOS (Homebrew)
brew install chromaprint ffmpeg

# Windows
winget install --id AcoustID.Chromaprint -e
winget install --id Gyan.FFmpeg -e
```

- **`fpcalc`** (Chromaprint) powers the DUPLICATE/REMASTER pipeline. It
  bundles its own decoder, so no Python bindings to `libchromaprint` are
  needed — `audiotwin` shells out to the binary (Chromaprint no longer
  ships a Windows `.dll`). Override the lookup with the `AUDIOTWIN_FPCALC`
  environment variable.
- **`ffmpeg`** powers everything else: `audiotwin.audio.decode_audio`
  streams PCM from ffmpeg entirely in memory (no converted files on disk).
  Override with `AUDIOTWIN_FFMPEG`.

## Quick start

```python
from audiotwin import detect

result = detect("track_a.mp3", "track_b.flac")
print(result["is_duplicate"], result["confidence"])
```

### CLI

```bash
audiotwin compare a.mp3 b.mp3 --json               # DUPLICATE pipeline
audiotwin classify a.mp3 b.mp3 --json              # DUPLICATE / REMASTER / NO_RELATION
audiotwin classify-edit matches.json --query-duration 180 --ref-duration 245
audiotwin landmark add index.db my_track track.mp3
audiotwin landmark query index.db query.mp3 [--pitch-range 2] [--json]
audiotwin cover compare a.mp3 b.mp3 --json
audiotwin fingerprint track.mp3
```

## The DUPLICATE pipeline — three levels

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

## Classifying relations (DUPLICATE vs REMASTER)

Beyond the binary duplicate/not-duplicate question, `classify_relation(...)`
(and its file-based counterpart `detect_relation(...)`) reads the *same two
scores* — `chromaprint_score` and `nfp_score` — through a finer grid, to also
recognize **REMASTER**: the same performance/recording, but with the signal
reworked (EQ, dynamics, a light re-mix). In one sentence: **REMASTER is the
signature of NFP staying high while Chromaprint drops** — same structural
content, different spectral texture.

| `chromaprint_score`                                 | `nfp_score`                       | `relation_type` |
| ---------------------------------------------------- | ---------------------------------- | ---------------- |
| `≥ duplicate_threshold` (0.85)                        | any                                 | `DUPLICATE`       |
| `remaster_chromaprint_min` (0.60) `≤ … <` `duplicate_threshold` | `≥ remaster_nfp_threshold` (0.90) | `REMASTER`        |
| `remaster_chromaprint_min` (0.60) `≤ … <` `duplicate_threshold` | `< remaster_nfp_threshold` or absent | `NO_RELATION`     |
| `< remaster_chromaprint_min` (0.60)                   | any                                 | `NO_RELATION`     |

A file hash match still short-circuits everything, straight to `DUPLICATE`
with `confidence = 1.0`.

```python
from audiotwin import detect_relation

result = detect_relation("track_a.mp3", "track_b_remaster.mp3", nfp_score=0.95)
print(result["relation_type"], result["score_gap"])  # "REMASTER" 0.25
```

Every threshold in the grid is a keyword parameter with the defaults shown
above.

## Classifying EDIT relations (speed change, trim, extend)

`classify_edit(...)` recognizes **temporal-structure edits**: cuts (radio
edit), additions (extended version), and uniform speed changes
(sped-up/slowed/nightcore — tempo and pitch moving together).

This function **never decodes audio**. It takes match points already
computed by an external segment-matching system — any segment-level
fingerprinter that outputs time-aligned correspondences works (e.g. a
neural fingerprinter in the spirit of NFP, Chang et al. 2021; there is no
dependency on any specific one). It fits `t_ref = slope × t_query +
intercept` with RANSAC and reads the relation off the geometry: the slope
is the speed factor, the coverage says how much of each track participates.

**The landmark module produces this exact format**: `match_points` from
`LandmarkIndex.query()` feed directly into `classify_edit()` /
`fit_temporal_alignment()`.

### Input format

Each match point is a `(t_query, t_ref, match_score)` triple. As JSON (for
the CLI):

```json
[
  [0.0,  30.0, 0.98],
  [5.0,  35.1, 0.95],
  [10.0, 40.0, 0.97],
  [15.0, 44.9, 0.96],
  [20.0, 50.1, 0.99],
  [25.0, 55.0, 0.94]
]
```

### Decision grid

| Fit result | Coverage | `edit_type_hint` |
|---|---|---|
| failed (too few points, no consistent line, slope out of bounds) | — | `no_relation` |
| `\|slope − 1\| >` `speed_change_epsilon` (0.03) | any | `speed_change` |
| slope ≈ 1 | either side `<` `full_coverage_threshold` (0.90) | `trim_or_extend` |
| slope ≈ 1 | both sides `≥` `full_coverage_threshold` | `full_match` |

`full_match` means **no edit detected — the temporal structure is intact**.
That pair is DUPLICATE/REMASTER territory: corroborate it with
`classify_relation()` rather than treating it as an edit.

**Why no scikit-learn?** RANSAC is implemented by hand with numpy.
`sklearn.linear_model.RANSACRegressor` would have pulled in scipy — measured
at ~46 MB of additional wheels — which conflicts with the core's
install-in-seconds goal. For fitting a 2-parameter line, the manual
implementation is ~40 lines and dependency-free. (The `[landmark]` extra
does bring scipy in, but only for users who opt into that module.)

## Landmark fingerprinting (SAMPLE and MASHUP) — `[landmark]` extra

`audiotwin.landmark` implements spectral-peak fingerprinting in the spirit
of **Wang 2003** (the Shazam paper): STFT peaks paired into 32-bit hashes,
matched through an offset histogram. Robust to noise and additive mixing,
and returns the **exact position** of each correspondence.

```python
from audiotwin.landmark import LandmarkIndex, classify_sample, classify_mashup

index = LandmarkIndex("catalog.db")          # or ":memory:"
index.add_track("original", "original.mp3")

results = index.query("suspect.mp3")
top = results[0]
print(top["offset_seconds"], top["aligned_hashes"], top["score"])

# Is the match a localized fragment (SAMPLE) or a global relation?
sample = classify_sample(top, query_duration=210.0, ref_duration=180.0)

# Do several distinct indexed tracks cover disjoint regions (MASHUP)?
mashup = classify_mashup(results, query_duration=210.0)
```

- `query(..., pitch_shift_range=2)` also tries pitch-shifted variants
  (±1..±2 semitones; needs the `[cover]` extra for librosa) and reports
  `pitch_shift_semitones` per match.
- `match_points` in each result plugs straight into `classify_edit()`.

## Cover similarity (COVER) — `[cover]` extra

`audiotwin.cover` implements the classic chroma pipeline (a simplified
**Serrà 2009**): harmonic separation → CQT chroma → **Optimal
Transposition Index** (key alignment) → **DTW** (tempo absorption).

```python
from audiotwin.cover import cover_similarity

result = cover_similarity("original.mp3", "cover_version.mp3")
print(result["similarity"])               # 0.0–1.0
print(result["transposition_semitones"])  # how far B is transposed up vs A
print(result["duration_ratio"])           # free global tempo hint
```

Callers that cache chroma features can skip re-decoding with
`cover_similarity_from_chroma(chroma_a, chroma_b)` (identical scoring), and
compute features via `compute_chroma(audio, sr)`.

## Instrumental / karaoke pairs (INSTRUMENTAL)

`classify_instrumental_pair(...)` detects the pattern *same musical
content, one track has vocals, the other doesn't*. Pure score arithmetic,
core-only:

```python
from audiotwin import classify_instrumental_pair

r = classify_instrumental_pair(
    content_similarity=0.85,   # any [0,1] "same content" measure:
                               # cover similarity, landmark score, NFP...
    vocal_coverage_a=0.70,     # from an external vocal-activity detector
    vocal_coverage_b=0.03,
)
print(r["is_instrumental_pair"], r["vocal_track"])  # True "a"
```

audiotwin deliberately ships no vocal detector — feed it coverages from
whatever VAD you trust.

The companion library
[vocalcoverage](https://github.com/ZeWills987/vocalcoverage) was built
precisely to produce this input: given a mix and an already-separated vocal
stem, it measures per-frame vocal presence (RMS ratio + harmonic f0
confirmation via pyin) and returns a `vocal_coverage` in `[0, 1]`. The full
chain, with a separator of your choice upstream:

```python
from vocalcoverage import analyze
from audiotwin import classify_instrumental_pair
from audiotwin.cover import cover_similarity

# Stems produced upstream (Demucs, audio-separator, ...):
#   a_vocals.wav — vocal stem of track A
#   b_vocals.wav — vocal stem of track B

content = cover_similarity("track_a.mp3", "track_b.mp3")

r = classify_instrumental_pair(
    content_similarity=content["similarity"],
    vocal_coverage_a=analyze("track_a.mp3", "a_vocals.wav")["vocal_coverage"],
    vocal_coverage_b=analyze("track_b.mp3", "b_vocals.wav")["vocal_coverage"],
)
```

## Neural signal (NFP) — `[neural]` extra

Everywhere audiotwin accepts an "NFP score" from an external system,
`audiotwin.neural` can now be that system: it wraps Sony's **Sample-ID**
model (Riou, Serrà & Mitsufuji, ICASSP 2026 — MIT license, code *and*
Zenodo checkpoint; downloaded on first use, never redistributed).

```python
from audiotwin import detect_relation
from audiotwin.neural import neural_similarity, neural_match_points

# 1) The missing REMASTER signal — plugs straight into detect()/detect_relation():
nfp = neural_similarity("track_a.mp3", "track_b.mp3")
verdict = detect_relation("track_a.mp3", "track_b.mp3", nfp_score=nfp["nfp_score"])

# 2) A transformation-robust matcher for classify_edit() — survives the
#    EQ/overdubs/speed edits that break landmark match points:
from audiotwin import classify_edit
points = neural_match_points("edited.mp3", "original.mp3")
edit = classify_edit(points, query_duration=180.0, ref_duration=245.0)
```

`neural_similarity` returns exactly the `nfp_score` / `nfp_segments_matched`
/ `nfp_coverage` triple that `detect`, `detect_relation` and
`combine_scores` accept; `neural_match_points` returns the
`(t_query, t_ref, score)` triples that `classify_edit` consumes. CPU
inference works (no GPU required); the model chunks tracks into 5 s
windows with 2.5 s hop.

## Aggregating everything: `suggest_relation()`

```python
from audiotwin import suggest_relation

verdict = suggest_relation(
    chromaprint_score=0.72,
    nfp_score=0.95,
    cover_result=cover_result,        # all inputs optional
)
for h in verdict["hypotheses"]:
    print(h["relation"], h["confidence"], h["evidence"])
```

Hypotheses are ordered by the documented priority (`DUPLICATE > REMASTER >
MASHUP > SAMPLE > EDIT > INSTRUMENTAL > COVER`), each carrying its source
confidence.

> **This is a rule-based convenience heuristic, not a trained classifier.**
> Production systems should train their own fusion on their own data.

## The "stems" pattern

Every function in this toolkit takes *any* audio file — including **stems**
produced by an external source separator (e.g. audio-separator or Demucs;
audiotwin has no dependency on either).

This matters for remix detection: a remix that keeps the original vocals
over a brand-new instrumentation often **fails** full-mix matching (the
instrumentation dominates the spectrum), but succeeds when you fingerprint
the **isolated vocal stems**:

```python
from audiotwin.landmark import LandmarkIndex

# Stems produced upstream by your separator of choice:
#   original_vocals.wav   (from the original track)
#   suspect_vocals.wav    (from the suspected remix)

index = LandmarkIndex(":memory:")
index.add_track("original_vocals", "original_vocals.wav")
matches = index.query("suspect_vocals.wav")
# A strong offset-consistent match on vocal stems + a weak full-mix match
# is the classic signature of a remix reusing the original vocal take.
```

The same applies to `cover_similarity` on instrumental stems (chord
progression without vocal interference), or `classify_edit` on match points
computed from stems.

## Limitations

Honest per-module limits:

- **DUPLICATE/REMASTER (Chromaprint + NFP)**
  - Detects the *same recording* only — covers/live/remixes score low by
    design. ≥ 10 seconds of audio required (`AudioTooShortError` below).
  - Without a caller-supplied NFP score, precision is slightly lower
    (~98% vs ~99.5% with NFP) and REMASTER cannot be distinguished from an
    unrelated near-miss.
- **EDIT (`classify_edit`)**
  - `edit_type_hint` is a geometric signal, not a final verdict —
    distinguishing radio edit from extended version, or nightcore from
    generic speed-up, needs context (title parsing, duration comparison)
    this library intentionally does not attempt. Quality depends entirely
    on the caller-supplied match points.
- **landmark (SAMPLE/MASHUP)**
  - Not robust to pitch shifting or time stretching by itself;
    `pitch_shift_range` compensates but remains a coarse ±N-semitone grid
    with no time-stretch handling.
  - Heavily overdubbed samples (e.g. a drum break buried under new layers)
    are hard to catch — additive-mixing robustness has limits.
  - The landmark family generally underperforms neural approaches on
    strongly transformed audio.
- **cover**
  - A classical method — precision is below recent deep-learning cover
    detectors. Microtonality and complex polyrhythms are not handled. The
    score is only as good as the chroma features.
- **suggest_relation**
  - A rule-based heuristic, not a trained model.

### What audiotwin deliberately does not implement

Neural audio fingerprinting and learned sample identification outperform
the classical methods here on transformed audio; they also carry model
weights and GPU-sized dependencies that don't belong in this toolkit.
audiotwin gives you the classical, dependency-light baselines — and clean
integration points (NFP scores, match points, content similarities) to
plug neural systems in when you have them. References: Wang 2003 (landmark
fingerprinting), Serrà 2009 (cover detection), Chang et al. 2021 (neural
fingerprinting).

## Scope

`audiotwin` returns analysis data — nothing more. It does **not** do source
separation, transcription, vocal detection, canonical-track selection, or
duplicate merging. Those decisions belong to the calling application.

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Built to power duplicate detection at Mkzik.</sub>
