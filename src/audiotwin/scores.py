"""Raw scores extraction — signals only, no thresholds, no verdicts.

This module is the "pure scores" counterpart of the ``classify_*`` /
``suggest_relation`` convenience layer: it exposes every numeric signal
audiotwin can compute about a pair of tracks, in a flat JSON-serializable
dict, WITHOUT applying any threshold or emitting any verdict. It is meant
for production pipelines that train their own decision layer (XGBoost,
random forest, neural fusion, ...) on top of the signals.

Contract:

* :func:`extract_all_scores` never calls ``classify_*``,
  ``suggest_relation`` or anything that applies a decision threshold —
  only the underlying computation functions.
* Missing optional extras are NOT an error: the corresponding fields are
  simply absent from the returned dict (one warning is logged per missing
  extra per process).
* The returned dict is flat and ``json.dumps()``-able: native ``float`` /
  ``int`` / ``bool`` / ``list`` only, no numpy types.
"""

from __future__ import annotations

import logging

from audiotwin.core import compare_fingerprints, compute_fingerprint, file_hash

logger = logging.getLogger(__name__)

_WARNED: set[str] = set()


def _warn_once(extra: str, detail: str) -> None:
    if extra not in _WARNED:
        _WARNED.add(extra)
        logger.warning(
            "extract_all_scores: extra [%s] indisponible (%s) — champs %s_* omis",
            extra,
            detail,
            extra,
        )


def _match_points_to_lists(points) -> list[list[float]]:
    return [[float(t_q), float(t_r), float(s)] for t_q, t_r, s in points]


def _chromaprint_scores(path_a: str, path_b: str) -> dict:
    scores: dict = {"file_hash_match": file_hash(path_a) == file_hash(path_b)}
    fp_a = compute_fingerprint(path_a)
    fp_b = compute_fingerprint(path_b)
    scores["chromaprint_score"] = float(compare_fingerprints(fp_a, fp_b))
    return scores


def _landmark_scores(path_a: str, path_b: str) -> dict:
    from audiotwin.landmark import LandmarkIndex

    index = LandmarkIndex(":memory:")
    index.add_track("b", path_b)
    # min_aligned_hashes=1: report the best offset bin even for a single
    # aligned hash — filtering is the caller's decision, not ours.
    results = index.query(path_a, min_aligned_hashes=1)
    if not results:
        return {
            "landmark_aligned_hashes": 0,
            "landmark_score": 0.0,
            "landmark_offset_seconds": None,
            "landmark_match_points": [],
        }
    top = results[0]
    return {
        "landmark_aligned_hashes": int(top["aligned_hashes"]),
        "landmark_score": float(top["score"]),
        "landmark_offset_seconds": float(top["offset_seconds"]),
        "landmark_match_points": _match_points_to_lists(top["match_points"]),
    }


def _cover_scores(path_a: str, path_b: str, include_embeddings: bool) -> dict:
    from audiotwin.audio import decode_audio
    from audiotwin.cover import (
        DEFAULT_COVER_SR,
        compute_chroma,
        cover_embedding,
        cover_similarity,
    )

    result = cover_similarity(path_a, path_b)
    scores = {
        "cover_similarity": float(result["similarity"]),
        "cover_transposition_semitones": int(result["transposition_semitones"]),
        "cover_dtw_normalized_cost": float(result["dtw_normalized_cost"]),
        "cover_duration_ratio": float(result["duration_ratio"]),
    }
    if include_embeddings:
        chroma_a = compute_chroma(decode_audio(path_a, sr=DEFAULT_COVER_SR))
        chroma_b = compute_chroma(decode_audio(path_b, sr=DEFAULT_COVER_SR))
        scores["cover_embedding_a"] = [float(x) for x in cover_embedding(chroma_a)]
        scores["cover_embedding_b"] = [float(x) for x in cover_embedding(chroma_b)]
    return scores


def _neural_scores(path_a: str, path_b: str) -> dict:
    import numpy as np

    from audiotwin.neural import (
        DEFAULT_CHUNK_HOP_SECONDS,
        DEFAULT_CHUNK_SECONDS,
        DEFAULT_COSINE_FLOOR,
        _calibrate,
        neural_embedding,
    )

    emb_a = neural_embedding(path_a)
    emb_b = neural_embedding(path_b)
    raw_matrix = emb_a @ emb_b.T
    best_raw = raw_matrix.max(axis=1)
    best_calibrated = _calibrate(best_raw, DEFAULT_COSINE_FLOOR)

    # ALL chunk correspondences with their scores — no match threshold:
    # filtering (and RANSAC via fit_temporal_alignment) is the caller's
    # decision layer, not ours.
    center = DEFAULT_CHUNK_SECONDS / 2.0
    points = []
    for i in range(raw_matrix.shape[0]):
        j = int(raw_matrix[i].argmax())
        points.append(
            [
                float(i * DEFAULT_CHUNK_HOP_SECONDS + center),
                float(j * DEFAULT_CHUNK_HOP_SECONDS + center),
                float(best_calibrated[i]),
            ]
        )

    return {
        "neural_similarity": float(np.mean(best_calibrated)),
        "neural_similarity_raw": float(np.mean(best_raw)),
        "neural_match_points": points,
    }


def _vocal_scores(path_a: str, path_b: str, stem_a: str, stem_b: str) -> dict:
    from vocalcoverage import analyze

    return {
        "vocal_coverage_a": float(analyze(path_a, stem_a)["vocal_coverage"]),
        "vocal_coverage_b": float(analyze(path_b, stem_b)["vocal_coverage"]),
    }


def extract_all_scores(
    path_a: str,
    path_b: str,
    include_chromaprint: bool = True,
    include_landmark: bool = True,
    include_cover: bool = True,
    include_neural: bool = False,
    include_vocal: bool = False,
    vocal_stem_a: str | None = None,
    vocal_stem_b: str | None = None,
    include_embeddings: bool = False,
) -> dict:
    """Every raw signal about a pair of tracks — no thresholds, no verdicts.

    The counterpart of the ``classify_*`` convenience layer for production
    pipelines that build their OWN decision logic (XGBoost, neural fusion,
    hand-tuned rules...): a flat, JSON-serializable dict of numeric
    signals, computed by calling the underlying functions directly
    (:func:`audiotwin.core.compare_fingerprints`,
    :meth:`audiotwin.landmark.LandmarkIndex.query`,
    :func:`audiotwin.cover.cover_similarity`, the Sample-ID embeddings...)
    and NEVER the decision layer.

    Field notes (only present when the section is enabled AND its extra is
    installed — a missing extra logs one warning and omits its fields):

    * ``landmark_*``: the index is queried with ``min_aligned_hashes=1``
      so even a 1-hash best bin is reported — filtering is your decision.
    * ``*_match_points``: ``[t_query, t_ref, score]`` triples, the direct
      input of :func:`audiotwin.core.fit_temporal_alignment` /
      :func:`classify_edit` if you want to run the RANSAC geometry with
      your own thresholds. ``neural_match_points`` carries ALL chunk
      correspondences (no match threshold), scored on the calibrated
      scale.
    * ``neural_similarity`` is the calibrated score (see
      :data:`audiotwin.neural.DEFAULT_COSINE_FLOOR`);
      ``neural_similarity_raw`` is the untouched mean best cosine.
    * ``vocal_coverage_*`` requires the separately-installed
      ``vocalcoverage`` package and pre-separated vocal stems.
    * Embeddings (``cover_embedding_*``) are excluded by default
      (``include_embeddings=False``) — they are for callers building
      external indexes.

    Args:
        path_a: First audio file (the "query" side for landmark/neural).
        path_b: Second audio file (the "reference" side).
        include_chromaprint: Level-0/1 signals (default True).
        include_landmark: Landmark signals (default True; needs [landmark]).
        include_cover: Chroma/DTW signals (default True; needs [cover]).
        include_neural: Sample-ID signals (default False — expensive;
            needs [neural] + the sampleid package).
        include_vocal: Vocal coverages (default False; needs the
            vocalcoverage package and both stems).
        vocal_stem_a: Pre-separated vocal stem of ``path_a``.
        vocal_stem_b: Pre-separated vocal stem of ``path_b``.
        include_embeddings: Also return fixed-size embeddings
            (default False).

    Returns:
        A flat dict of native-Python values, ``json.dumps()``-able.
    """
    scores: dict = {}

    if include_chromaprint:
        scores.update(_chromaprint_scores(path_a, path_b))

    if include_landmark:
        try:
            scores.update(_landmark_scores(path_a, path_b))
        except ImportError as exc:
            _warn_once("landmark", str(exc))

    if include_cover:
        try:
            scores.update(_cover_scores(path_a, path_b, include_embeddings))
        except ImportError as exc:
            _warn_once("cover", str(exc))

    if include_neural:
        try:
            scores.update(_neural_scores(path_a, path_b))
        except ImportError as exc:
            _warn_once("neural", str(exc))

    if include_vocal:
        if vocal_stem_a is None or vocal_stem_b is None:
            raise ValueError(
                "include_vocal=True exige vocal_stem_a ET vocal_stem_b "
                "(stems vocaux pré-séparés)"
            )
        try:
            scores.update(_vocal_scores(path_a, path_b, vocal_stem_a, vocal_stem_b))
        except ImportError as exc:
            _warn_once("vocal", str(exc))

    return scores
