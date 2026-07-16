"""Neural fingerprinting bridge — Sony Sample-ID embeddings.

Wraps the MIT-licensed Sample-ID model (Riou, Serrà & Mitsufuji, ICASSP
2026; https://github.com/sony/sampleid) as the neural signal the rest of
audiotwin was designed to accept but never computes itself:

* :func:`neural_similarity` returns exactly the ``nfp_score`` /
  ``nfp_segments_matched`` / ``nfp_coverage`` triple that
  :func:`audiotwin.core.detect`, :func:`audiotwin.core.detect_relation`
  and :func:`audiotwin.core.combine_scores` take as optional inputs —
  unlocking REMASTER classification without any external system.
* :func:`neural_match_points` returns ``(t_query, t_ref, score)`` triples
  in the exact format :func:`audiotwin.core.classify_edit` and
  :func:`audiotwin.core.fit_temporal_alignment` consume — a matcher that
  survives the transformations (EQ, overdubs, pitch/time edits) that
  break the classical landmark front-end.

Requires the ``[neural]`` extra (PyTorch) plus the ``sampleid`` package
installed from GitHub (it is not on PyPI)::

    pip install "audiotwin[neural]"
    pip install -e "git+https://github.com/sony/sampleid.git#egg=sampleid"

The pretrained checkpoint (MIT, Zenodo record 17413869) is downloaded by
audiotwin on first use into ``~/.cache/audiotwin`` (override with the
``AUDIOTWIN_CACHE`` environment variable) — audiotwin redistributes
neither the model code nor the weights.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

from audiotwin.audio import decode_audio
from audiotwin.core import compute_coverage, fit_temporal_alignment

#: Sample rate the Sample-ID model expects.
NEURAL_SR = 16000

#: Default chunk length in seconds (the model averages embeddings over its
#: input, so full-track similarity needs manual chunking).
DEFAULT_CHUNK_SECONDS = 5.0

#: Default hop between chunks in seconds (50% overlap).
DEFAULT_CHUNK_HOP_SECONDS = 2.5

#: Default CALIBRATED similarity above which a query chunk counts as matched.
DEFAULT_MATCH_THRESHOLD = 0.60

#: Default calibrated-score floor for LOCALIZED matching. Deliberately much
#: lower than DEFAULT_MATCH_THRESHOLD: a short, transformed fragment
#: (re-pitched sample, vocals over a new instrumentation) scores well below
#: whole-track duplicates, and the false positives a low threshold lets in
#: are killed downstream by the RANSAC temporal-coherence test instead.
DEFAULT_LOCALIZED_THRESHOLD = 0.25

#: Raw-cosine floor of the calibration. Sample-ID's embedding space is
#: compressed: on real music, UNRELATED tracks already sit around 0.95 raw
#: cosine while same-master pairs sit at ~0.999 (measured on real
#: SoundCloud/YouTube pairs). Raw cosines would therefore wrongly clear
#: detect()'s nfp thresholds for any pair; all scores are rescaled as
#: ``(cos - floor) / (1 - floor)`` (clamped to [0, 1]) so unrelated ≈ 0 and
#: same-master ≈ 0.98. If your catalog's genre mix differs, measure the
#: mean best-cosine of a few unrelated pairs and pass it as
#: ``cosine_floor``.
DEFAULT_COSINE_FLOOR = 0.95

#: Default batch size for embedding extraction.
DEFAULT_BATCH_SIZE = 16

#: Override the checkpoint cache directory via this environment variable.
CACHE_DIR_ENVVAR = "AUDIOTWIN_CACHE"

_MODEL_CACHE: dict = {}


def _cache_dir() -> str:
    return os.environ.get(
        CACHE_DIR_ENVVAR, os.path.join(os.path.expanduser("~"), ".cache", "audiotwin")
    )


def _default_checkpoint(SampleID) -> str:
    """Download (once) the official Zenodo checkpoint into audiotwin's cache.

    audiotwin handles the download itself instead of delegating to
    ``sampleid.load_checkpoint()``: upstream's downloader moves a still-open
    ``NamedTemporaryFile(delete=True)`` into place, and on Windows the
    delete-on-close flag follows the handle — the moved checkpoint vanishes
    when the tempfile closes. Downloading to ``<target>.part`` then
    ``os.replace`` is atomic and cross-platform, and keeps the 805 MB file
    in a user cache instead of site-packages.
    """
    target = os.path.join(_cache_dir(), "sampleid-best.ckpt")
    if os.path.exists(target):
        return target

    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".part"
    urllib.request.urlretrieve(SampleID.default_ckpt_url, partial)
    os.replace(partial, target)
    return target


def _require_sampleid():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the neural module requires PyTorch — install it with "
            "'pip install audiotwin[neural]'"
        ) from exc
    try:
        from sampleid import SampleID
    except ImportError as exc:
        raise ImportError(
            "the neural module requires the 'sampleid' package (MIT, not on "
            "PyPI) — install it in EDITABLE mode (upstream packaging omits "
            "its src/ subpackage in regular installs): pip install -e "
            '"git+https://github.com/sony/sampleid.git#egg=sampleid"'
        ) from exc
    return SampleID


def _load_model(checkpoint_path: str | None = None):
    """Load (and memoize) the Sample-ID model in eval mode."""
    import torch

    SampleID = _require_sampleid()
    key = checkpoint_path or "__default__"
    if key not in _MODEL_CACHE:
        resolved = checkpoint_path or _default_checkpoint(SampleID)
        model = SampleID.load_checkpoint(ckpt_path=resolved)
        model.eval()
        torch.set_grad_enabled(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def neural_embedding(
    path: str,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_hop_seconds: float = DEFAULT_CHUNK_HOP_SECONDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_path: str | None = None,
) -> np.ndarray:
    """Per-chunk Sample-ID embeddings for an audio file.

    The track is decoded to 16 kHz mono in memory, split into overlapping
    chunks, and each chunk is embedded independently (the model averages
    over its input, so chunking is what preserves temporal resolution).

    Args:
        path: Path to the audio file.
        chunk_seconds: Chunk length (default 5.0 s, the model's native
            training length).
        chunk_hop_seconds: Hop between chunk starts (default 2.5 s).
        batch_size: Chunks embedded per forward pass (default 16).
        checkpoint_path: Optional custom checkpoint; default downloads the
            official Zenodo checkpoint on first use.

    Returns:
        A ``(n_chunks, embed_dim)`` float32 array of L2-normalized
        embeddings, in temporal order. Chunk ``i`` starts at
        ``i * chunk_hop_seconds`` seconds.
    """
    audio = decode_audio(path, sr=NEURAL_SR)
    return _embed_audio(audio, chunk_seconds, chunk_hop_seconds, batch_size, checkpoint_path)


def _embed_audio(
    audio: np.ndarray,
    chunk_seconds: float,
    chunk_hop_seconds: float,
    batch_size: int,
    checkpoint_path: str | None,
) -> np.ndarray:
    """Embed an in-memory PCM buffer (16 kHz mono) per chunk."""
    import torch

    model = _load_model(checkpoint_path)

    chunk_len = int(chunk_seconds * NEURAL_SR)
    hop_len = int(chunk_hop_seconds * NEURAL_SR)
    if len(audio) < chunk_len:
        audio = np.pad(audio, (0, chunk_len - len(audio)))

    starts = list(range(0, len(audio) - chunk_len + 1, hop_len))
    chunks = np.stack([audio[s : s + chunk_len] for s in starts])

    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(chunks), batch_size):
            batch = torch.from_numpy(chunks[i : i + batch_size])
            out = model(batch, audio=True)  # (batch, 1, embed_dim)
            embeddings.append(out.squeeze(1).cpu().numpy())

    matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _calibrate(raw_cosine: np.ndarray, cosine_floor: float) -> np.ndarray:
    """Rescale raw Sample-ID cosines to [0, 1] (see DEFAULT_COSINE_FLOOR)."""
    return np.clip((raw_cosine - cosine_floor) / (1.0 - cosine_floor), 0.0, 1.0)


def neural_similarity(
    path_a: str,
    path_b: str,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_hop_seconds: float = DEFAULT_CHUNK_HOP_SECONDS,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    checkpoint_path: str | None = None,
) -> dict:
    """Neural content similarity between two files, in detect()'s NFP format.

    Each chunk of A is compared (cosine) against every chunk of B, cosines
    are CALIBRATED (see :data:`DEFAULT_COSINE_FLOOR` — raw Sample-ID
    cosines live in a compressed range where unrelated music already scores
    ~0.95); a chunk "matches" when its best calibrated score clears
    ``match_threshold``.

    Returns a dict whose keys are named to plug STRAIGHT into
    :func:`audiotwin.core.detect` / :func:`detect_relation` /
    :func:`combine_scores`::

        nfp = neural_similarity(a, b)
        verdict = detect(a, b, nfp_score=nfp["nfp_score"],
                         nfp_segments_matched=nfp["nfp_segments_matched"],
                         nfp_coverage=nfp["nfp_coverage"])

    Args:
        path_a: Query track path.
        path_b: Reference track path.
        match_threshold: Cosine floor for a chunk to count as matched
            (default 0.60).
        chunk_seconds: See :func:`neural_embedding`.
        chunk_hop_seconds: See :func:`neural_embedding`.
        checkpoint_path: Optional custom checkpoint.

    Returns:
        A dict with ``nfp_score`` (mean over A's chunks of the best
        CALIBRATED similarity against B, in [0, 1] — measured anchors:
        same-master pairs ≈ 0.98, unrelated ≈ 0.0), ``nfp_segments_matched``
        (number of A-chunks above the threshold) and ``nfp_coverage`` (that
        count over A's total chunks).
    """
    emb_a = neural_embedding(
        path_a,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )
    emb_b = neural_embedding(
        path_b,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )

    similarity_matrix = emb_a @ emb_b.T  # (chunks_a, chunks_b), raw cosine
    best_per_chunk = _calibrate(similarity_matrix.max(axis=1), cosine_floor)

    matched = int((best_per_chunk >= match_threshold).sum())
    return {
        "nfp_score": float(best_per_chunk.mean()),
        "nfp_segments_matched": matched,
        "nfp_coverage": matched / len(best_per_chunk),
    }


def neural_match_points(
    path_a: str,
    path_b: str,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_hop_seconds: float = DEFAULT_CHUNK_HOP_SECONDS,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    checkpoint_path: str | None = None,
) -> list[tuple[float, float, float]]:
    """Chunk-level correspondences in classify_edit()'s match-point format.

    For each chunk of A whose best CALIBRATED similarity against B (see
    :data:`DEFAULT_COSINE_FLOOR`) clears the threshold, emit
    ``(t_query, t_ref, score)`` at the two chunks' center times. The
    output feeds :func:`audiotwin.core.classify_edit` /
    :func:`fit_temporal_alignment` directly — a transformation-robust
    alternative to landmark match points (the embeddings survive EQ,
    overdubs and moderate pitch/time edits that break spectral-peak
    hashing)::

        points = neural_match_points(edited.mp3, original.mp3)
        verdict = classify_edit(points, query_duration=..., ref_duration=...)

    Args:
        path_a: Query track path.
        path_b: Reference track path.
        match_threshold: Cosine floor for emitting a point (default 0.60).
        chunk_seconds: See :func:`neural_embedding`.
        chunk_hop_seconds: See :func:`neural_embedding`.
        checkpoint_path: Optional custom checkpoint.

    Returns:
        ``(t_query, t_ref, score)`` triples, one per matched A-chunk,
        sorted by ``t_query``.

    Note:
        Points sit at CHUNK CENTERS, so they can never reach the track
        edges: maximum achievable coverage is
        ``(duration - chunk_seconds) / duration`` (e.g. 0.83 for a 30 s
        track with 5 s chunks). When feeding
        :func:`audiotwin.core.classify_edit`, lower
        ``full_coverage_threshold`` accordingly or use longer tracks.
    """
    emb_a = neural_embedding(
        path_a,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )
    emb_b = neural_embedding(
        path_b,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )

    similarity_matrix = _calibrate(emb_a @ emb_b.T, cosine_floor)
    center = chunk_seconds / 2.0

    points = []
    for i in range(similarity_matrix.shape[0]):
        j = int(similarity_matrix[i].argmax())
        score = float(similarity_matrix[i, j])
        if score >= match_threshold:
            points.append(
                (
                    i * chunk_hop_seconds + center,
                    j * chunk_hop_seconds + center,
                    score,
                )
            )
    return points


def neural_localized_match(
    path_a: str,
    path_b: str,
    match_threshold: float = DEFAULT_LOCALIZED_THRESHOLD,
    min_inliers: int = 4,
    residual_threshold: float = 2.6,
    pitch_shift_range: int = 0,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_hop_seconds: float = DEFAULT_CHUNK_HOP_SECONDS,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    checkpoint_path: str | None = None,
) -> dict:
    """Find a LOCALIZED aligned fragment of B inside A (sample / kept stem).

    Whole-track scores dilute a short match: a 10 s sample inside a 3 min
    track barely moves ``neural_similarity``'s mean, and a re-pitched or
    overdubbed fragment scores below :data:`DEFAULT_MATCH_THRESHOLD` per
    chunk. This function inverts the strategy — accept LOW-scoring chunk
    correspondences (:data:`DEFAULT_LOCALIZED_THRESHOLD`), then demand that
    the surviving points form a temporally coherent line
    (:func:`audiotwin.core.fit_temporal_alignment`, RANSAC): random
    false-positive chunks don't align, a genuine fragment does.

    This is the intended tool for the cases full-track matching misses:
    short re-pitched samples, mashup sources, and remixes that keep the
    original vocals (run it on separated vocal stems for the latter — the
    "stems pattern" in the README).

    Args:
        path_a: Query track (the one suspected of CONTAINING the fragment).
        path_b: Reference track (the fragment's origin).
        match_threshold: Calibrated per-chunk score floor (default 0.25 —
            deliberately permissive, RANSAC filters the noise).
        min_inliers: Minimum temporally coherent points (default 4, i.e.
            ~10 s of matched material at the default hop).
        residual_threshold: RANSAC inlier tolerance in seconds (default
            2.6 ≈ one chunk hop; chunk centers quantize the timeline).
        pitch_shift_range: 0 disables (default); N also tries the query
            pitch-shifted by ±1..±N semitones and keeps the best alignment
            (Sample-ID embeddings are not pitch-invariant: a fragment
            re-pitched by even 1 semitone drops below any usable
            threshold). Requires librosa (the ``[cover]`` extra). The
            winning shift is reported as ``pitch_shift_semitones``.
        chunk_seconds: See :func:`neural_embedding`.
        chunk_hop_seconds: See :func:`neural_embedding`.
        cosine_floor: See :data:`DEFAULT_COSINE_FLOOR`.
        checkpoint_path: Optional custom checkpoint.

    Returns:
        A dict with ``found`` (bool), ``slope`` (speed factor),
        ``offset_seconds`` (intercept), ``inlier_count``,
        ``match_start_query`` / ``match_end_query`` / ``match_start_ref`` /
        ``match_end_ref`` (fragment bounds, ``None`` when not found),
        ``coverage_query`` / ``coverage_ref`` and ``confidence`` (mean
        calibrated score of the inlier points; 0.0 when not found).
    """
    audio_a = decode_audio(path_a, sr=NEURAL_SR)
    emb_b = neural_embedding(
        path_b,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )

    shifts = [0]
    if pitch_shift_range > 0:
        try:
            import librosa  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pitch_shift_range requires librosa — install it with "
                "'pip install audiotwin[cover]' (or audiotwin[all])"
            ) from exc
        shifts += [s for s in range(-pitch_shift_range, pitch_shift_range + 1) if s != 0]

    best = _NOT_FOUND_LOCALIZED.copy()
    for shift in shifts:
        if shift == 0:
            query_audio = audio_a
        else:
            import librosa

            query_audio = librosa.effects.pitch_shift(audio_a, sr=NEURAL_SR, n_steps=shift)
        emb_a = _embed_audio(
            query_audio, chunk_seconds, chunk_hop_seconds, DEFAULT_BATCH_SIZE, checkpoint_path
        )
        result = _localized_match_from_embeddings(
            emb_a,
            emb_b,
            match_threshold=match_threshold,
            min_inliers=min_inliers,
            residual_threshold=residual_threshold,
            chunk_seconds=chunk_seconds,
            chunk_hop_seconds=chunk_hop_seconds,
            cosine_floor=cosine_floor,
        )
        result["pitch_shift_semitones"] = shift
        if result["found"] and (
            not best["found"] or result["inlier_count"] > best["inlier_count"]
        ):
            best = result

    return best


_NOT_FOUND_LOCALIZED = {
    "found": False,
    "slope": 0.0,
    "offset_seconds": 0.0,
    "inlier_count": 0,
    "match_start_query": None,
    "match_end_query": None,
    "match_start_ref": None,
    "match_end_ref": None,
    "coverage_query": 0.0,
    "coverage_ref": 0.0,
    "confidence": 0.0,
    "pitch_shift_semitones": 0,
}


def _localized_match_from_embeddings(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    match_threshold: float,
    min_inliers: int,
    residual_threshold: float,
    chunk_seconds: float,
    chunk_hop_seconds: float,
    cosine_floor: float,
) -> dict:
    similarity_matrix = _calibrate(emb_a @ emb_b.T, cosine_floor)
    center = chunk_seconds / 2.0

    points: list[tuple[float, float, float]] = []
    for i in range(similarity_matrix.shape[0]):
        j = int(similarity_matrix[i].argmax())
        score = float(similarity_matrix[i, j])
        if score >= match_threshold:
            points.append(
                (i * chunk_hop_seconds + center, j * chunk_hop_seconds + center, score)
            )

    if len(points) < min_inliers:
        return _NOT_FOUND_LOCALIZED.copy()

    fit = fit_temporal_alignment(
        points,
        min_inliers=min_inliers,
        residual_threshold=residual_threshold,
    )
    if not fit["fit_succeeded"]:
        return _NOT_FOUND_LOCALIZED.copy()

    # Approximate durations from the chunk grids (± one hop).
    dur_a = (len(emb_a) - 1) * chunk_hop_seconds + chunk_seconds
    dur_b = (len(emb_b) - 1) * chunk_hop_seconds + chunk_seconds
    coverage = compute_coverage(points, fit["inlier_indices"], dur_a, dur_b)

    inliers = [points[k] for k in fit["inlier_indices"]]
    t_queries = [p[0] for p in inliers]
    t_refs = [p[1] for p in inliers]

    return {
        "found": True,
        "slope": fit["slope"],
        "offset_seconds": fit["intercept"],
        "inlier_count": fit["inlier_count"],
        "match_start_query": min(t_queries) - center,
        "match_end_query": max(t_queries) + center,
        "match_start_ref": min(t_refs) - center,
        "match_end_ref": max(t_refs) + center,
        "coverage_query": coverage["coverage_query"],
        "coverage_ref": coverage["coverage_ref"],
        "confidence": float(np.mean([p[2] for p in inliers])),
        "pitch_shift_semitones": 0,
    }
