# Changelog

## 0.4.1

- **neural: cosine calibration** — raw Sample-ID cosines live in a
  compressed range (measured on real cross-platform pairs: unrelated
  music ~0.95, same-master ~0.999), so raw scores would wrongly clear
  detect()'s nfp thresholds for ANY pair. All neural scores are now
  rescaled as `(cos - floor) / (1 - floor)` with a measured default
  floor of 0.95 (parametric via `cosine_floor`): same-master pairs now
  score ~0.98, unrelated ~0.0. Match thresholds operate on the
  calibrated scale.

## 0.4.0

- **New `[neural]` extra + `audiotwin.neural` module** — wraps Sony's
  Sample-ID model (Riou, Serrà & Mitsufuji, ICASSP 2026; MIT license for
  both code and the Zenodo checkpoint, downloaded on first use — nothing
  is redistributed). Deliberately NOT part of `[all]` (PyTorch is a
  ~2 GB install); also requires `pip install
  -e git+https://github.com/sony/sampleid.git#egg=sampleid` (editable install required — upstream packaging omits its src/ subpackage in regular installs; not on PyPI).
  - `neural_similarity()` produces the exact
    `nfp_score`/`nfp_segments_matched`/`nfp_coverage` triple that
    `detect()`/`detect_relation()`/`combine_scores()` accept — REMASTER
    classification no longer requires an external system.
  - `neural_match_points()` produces the `(t_query, t_ref, score)`
    triples that `classify_edit()`/`fit_temporal_alignment()` consume —
    a matcher robust to the EQ/overdub/speed transformations that break
    landmark match points.
  - `neural_embedding()` exposes the raw per-chunk embeddings (5 s
    windows, 2.5 s hop, L2-normalized) for callers building their own
    indexes.

## 0.3.0

Robustness pass on the weakest processes identified in the efficiency
review.

- **landmark: spectral whitening on by default** — each spectrogram
  column is divided by its smoothed spectral envelope before peak
  picking, so high-frequency and quiet-passage peaks compete fairly
  with the low-frequency energy mass. ⚠ **Extracted hashes change:
  rebuild existing LandmarkIndex databases** (or pass `whiten=False`
  on both index and query sides to keep the old behavior). A
  running-peak AGC (Stowell & Plumbley 2007) was evaluated first and
  rejected: it saturated bins to 1.0 and collapsed specificity.
- **landmark: adjacent offset-bin fusion** — a true offset landing near
  a histogram-bin boundary no longer splits its votes below the match
  threshold; the reported `offset_seconds` is now the mean of the
  matched pairs' actual offsets (more accurate than the bin center).
- **landmark: robust mashup regions** — `classify_mashup` region bounds
  use the 5th–95th percentiles of match-point times instead of min/max,
  so stray hash collisions can't stretch a region into its neighbors
  and veto a genuine source.
- **EDIT/RANSAC: score-weighted sampling** — hypothesis pairs are drawn
  proportionally to each point's `match_score` (previously ignored);
  disable with `weight_by_score=False`.
- **EDIT/RANSAC: adaptive termination** — sampling stops once the
  Fischler-Bolles iteration bound (`k = log(1-p)/log(1-w²)`) is
  reached, typically well under the 1000-iteration budget.
- **cover: 2DFTM pre-filter** — new `cover_embedding()` /
  `cover_embedding_similarity()` (Bertin-Mahieux & Ellis 2012): a
  fixed-size, transposition- and time-shift-invariant embedding for
  O(1)-per-pair candidate ranking before the exact DTW pipeline.
- **safety-net subprocess timeouts** — fpcalc and ffmpeg calls now fail
  loudly after 600 s instead of hanging forever on wedged decoders.

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
