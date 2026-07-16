"""Tests for audiotwin.scores — raw signals, no decisions.

The equality tests verify that extract_all_scores returns EXACTLY the
same numbers as the underlying functions called directly (no rounding, no
hidden transformation).
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from audiotwin.core import (
    FPCALC_COMMAND,
    FPCALC_COMMAND_ENVVAR,
    compare_fingerprints,
    compute_fingerprint,
    file_hash,
)
from audiotwin.scores import extract_all_scores

requires_chromaprint = pytest.mark.skipif(
    not shutil.which(os.environ.get(FPCALC_COMMAND_ENVVAR, FPCALC_COMMAND)),
    reason="Chromaprint (fpcalc) not installed",
)
requires_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@requires_chromaprint
@requires_ffmpeg
def test_values_match_individual_functions_exactly(sine_440, different_audio):
    scores = extract_all_scores(sine_440, different_audio, include_neural=False)

    # Chromaprint: bit-for-bit identical to the direct calls.
    assert scores["file_hash_match"] == (file_hash(sine_440) == file_hash(different_audio))
    direct = compare_fingerprints(
        compute_fingerprint(sine_440), compute_fingerprint(different_audio)
    )
    assert scores["chromaprint_score"] == direct  # equality, not approx

    # Cover: identical to cover_similarity called directly.
    librosa = pytest.importorskip("librosa")  # noqa: F841
    from audiotwin.cover import cover_similarity

    cov = cover_similarity(sine_440, different_audio)
    assert scores["cover_similarity"] == cov["similarity"]
    assert scores["cover_transposition_semitones"] == cov["transposition_semitones"]
    assert scores["cover_dtw_normalized_cost"] == cov["dtw_normalized_cost"]
    assert scores["cover_duration_ratio"] == cov["duration_ratio"]

    # Landmark: identical to a direct min_aligned_hashes=1 query.
    pytest.importorskip("scipy")
    from audiotwin.landmark import LandmarkIndex

    index = LandmarkIndex(":memory:")
    index.add_track("b", different_audio)
    results = index.query(sine_440, min_aligned_hashes=1)
    if results:
        assert scores["landmark_aligned_hashes"] == results[0]["aligned_hashes"]
        assert scores["landmark_score"] == results[0]["score"]
        assert scores["landmark_offset_seconds"] == results[0]["offset_seconds"]
    else:
        assert scores["landmark_aligned_hashes"] == 0


@requires_chromaprint
@requires_ffmpeg
def test_result_is_json_serializable(sine_440, sine_440_copy):
    scores = extract_all_scores(sine_440, sine_440_copy, include_embeddings=True)
    serialized = json.dumps(scores)  # must not raise
    assert isinstance(json.loads(serialized), dict)
    # Flat dict: no nested dicts (lists of numbers are allowed).
    assert not any(isinstance(v, dict) for v in scores.values())


@requires_chromaprint
def test_missing_extra_yields_partial_dict_not_exception(sine_440, different_audio, monkeypatch):
    import audiotwin.scores as scores_mod

    def boom(*args, **kwargs):
        raise ImportError("simulated missing extra")

    monkeypatch.setattr(scores_mod, "_landmark_scores", boom)
    monkeypatch.setattr(scores_mod, "_cover_scores", boom)
    scores_mod._WARNED.clear()

    scores = extract_all_scores(sine_440, different_audio)
    assert "chromaprint_score" in scores
    assert not any(k.startswith("landmark_") for k in scores)
    assert not any(k.startswith("cover_") for k in scores)


@requires_chromaprint
def test_no_decision_fields_ever(sine_440, sine_440_copy):
    scores = extract_all_scores(sine_440, sine_440_copy)
    forbidden = (
        "is_duplicate",
        "relation_type",
        "confidence",
        "edit_type_hint",
        "is_localized_match",
        "is_mashup_pattern",
        "is_instrumental_pair",
        "hypotheses",
    )
    assert not any(k in scores for k in forbidden)


def test_include_vocal_without_stems_raises(sine_440, sine_440_copy):
    with pytest.raises(ValueError, match="vocal_stem"):
        extract_all_scores(
            sine_440, sine_440_copy, include_chromaprint=False, include_landmark=False,
            include_cover=False, include_vocal=True,
        )


@requires_chromaprint
@requires_ffmpeg
def test_neural_fields_when_enabled(sine_440, sine_440_copy):
    pytest.importorskip("torch")
    pytest.importorskip("sampleid")

    scores = extract_all_scores(
        sine_440, sine_440_copy,
        include_landmark=False, include_cover=False, include_neural=True,
    )
    assert 0.0 <= scores["neural_similarity"] <= 1.0
    # Raw cosine lives above the calibrated score by construction.
    assert scores["neural_similarity_raw"] >= scores["neural_similarity"] - 1e-9
    # ALL chunks are reported (no threshold): one point per query chunk.
    assert len(scores["neural_match_points"]) > 0
    assert all(len(p) == 3 for p in scores["neural_match_points"])
    json.dumps(scores)  # still serializable with neural fields
