# Changelog

## 0.7.0

- **New `audiotwin.scores` module — raw signals, no decisions** —
  `extract_all_scores(path_a, path_b, ...)` returns a flat,
  JSON-serializable dict of every numeric signal (chromaprint, landmark
  with `min_aligned_hashes=1`, cover, calibrated + raw neural
  similarity, all-chunks match points, vocal coverages via the
  vocalcoverage package) WITHOUT applying any threshold or emitting any
  verdict. Contract: never calls the `classify_*` / `suggest_relation`
  decision layer; missing extras log one warning and omit their fields;
  embeddings excluded unless `include_embeddings=True`. This is the
  intended entry point for production pipelines that train their own
  decision layer; the `classify_*` conveniences are unchanged and now
  point to it in their docstrings.

## 0.6.0

- **suggest_relation: new REMIX hypothesis** — accepts a
  `remix_result` (the output of `neural_localized_match` run on
  separated VOCAL STEMS): a remix that keeps the original vocals over
  a new instrumentation is invisible to every full-mix signal but
  aligns on the stems. Priority slots between SAMPLE and EDIT.
  Validated on a real kept-vocals remix (stems aligned at slope 0.96
  with 11 coherent points while all full-mix signals were silent).

## 0.5.1

- **neural: GPU support** — the inference device is auto-detected
  (CUDA, then Apple MPS, else CPU) and can be forced per call via the
  new `device` parameter on every neural function, or globally via the
  `AUDIOTWIN_DEVICE` environment variable ("cuda", "cuda:1", "mps",
  "cpu"). On a GPU box with a CUDA build of torch installed, nothing
  else changes: the same code runs ~10-50x faster per pair.

## 0.5.0

- **neural: new `neural_localized_match()`** — finds a LOCALIZED aligned
  fragment of one track inside another (sample, mashup source, kept
  vocal stem). Whole-track scores dilute short matches, so this inverts
  the strategy: accept low-scoring chunk correspondences (default
  calibrated threshold 0.25) and let RANSAC's temporal-coherence test
  kill the false positives. Validated on real audio: a 20 s fragment
  overdub-mixed into another real track was located at the exact offset
  while whole-track nfp_score read 0.08. Optional `pitch_shift_range`
  compensation (librosa) for re-pitched fragments — with the honest
  caveat that heavily re-pitched samples under overdub remain hard even
  neurally (SOTA on Sample100 is mAP 0.603).

## 0.4.3

- **classify_relation: REMASTER NFP threshold recalibrated on real
  music** — with audiotwin.neural's calibrated scores, a true remaster
  measured nfp 0.83 (the reworked signal drifts the embedding), while
  covers/live measured <= 0.46 and unrelated < 0.06. The 0.90 default
  (tuned for same-master confirmation) wrongly rejected real
  remasters; `remaster_nfp_threshold` now defaults to 0.75, splitting
  the same-recording family from everything else with wide margins.
  detect()'s DUPLICATE confirmation keeps its stricter 0.90.

## 0.4.2

- **suggest_relation: cover threshold recalibrated on real music** —
  chroma-DTW similarity for UNRELATED real tracks already sits at
  ~0.82-0.84 (measured on cross-platform test pairs), so the previous
  0.60 default would have emitted a COVER hypothesis for any pair.
  `cover_similarity_threshold` now defaults to 0.85 (related versions
  measured 0.86-0.99). The margin is thin — an inherent limit of the
  classical chroma-DTW family.

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
