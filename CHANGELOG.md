# Changelog

## 0.2.0

Major extension: from duplicate detector to audio-relation toolkit.

- **`audiotwin.audio`** — shared in-memory ffmpeg decoding
  (`decode_audio`); no converted files ever written to disk.
- **`audiotwin.landmark`** (`[landmark]` extra, scipy) — spectral-peak
  fingerprinting (Wang 2003): `extract_landmarks`, SQLite-backed
  `LandmarkIndex` with offset-histogram matching and optional
  pitch-shift compensation (`pitch_shift_range`, needs librosa), plus
  `classify_sample` (localized SAMPLE) and `classify_mashup`
  (multi-source MASHUP).
- **`audiotwin.cover`** (`[cover]` extra, librosa) — cover-song
  similarity (simplified Serrà 2009): `compute_chroma`,
  `optimal_transposition` (OTI), `cover_similarity` and
  `cover_similarity_from_chroma`.
- **`classify_instrumental_pair`** — instrumental/karaoke pattern from
  caller-supplied content similarity + vocal coverages (core, no new
  dependency).
- **`suggest_relation`** — rule-based convenience aggregator producing
  ordered relation hypotheses (documented as a heuristic, not a trained
  classifier).
- CLI: new `landmark add`, `landmark query` (with `--pitch-range`,
  `--classify-sample`) and `cover compare` subcommands.
- Packaging: optional extras `[landmark]`, `[cover]`, `[all]`; heavy
  imports are lazy so the core install stays numpy-only.
- CI: test matrix now installs `[all]` + ffmpeg, plus a core-only job
  verifying that base `audiotwin` imports without scipy/librosa.

## 0.1.0

- Initial release: DUPLICATE detection (SHA256 file hash + Chromaprint
  via `fpcalc` + optional caller-supplied NFP score) with `detect()`
  and `combine_scores()`.
- REMASTER classification: `classify_relation()` / `detect_relation()`.
- EDIT classification from segment match points: `fit_temporal_alignment`
  (hand-rolled numpy RANSAC), `compute_coverage`, `classify_edit`.
- CLI: `compare`, `classify`, `classify-edit`, `fingerprint`.
