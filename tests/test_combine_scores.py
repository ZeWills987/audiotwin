"""Unit tests for the pure decision logic — no audio decoding involved."""

import pytest

from audiotwin import combine_scores


def test_hash_match_short_circuits():
    r = combine_scores(chromaprint_score=0.0, file_hash_match=True)
    assert r["is_duplicate"] is True
    assert r["confidence"] == 1.0


def test_hash_match_ignores_contradicting_nfp():
    r = combine_scores(chromaprint_score=0.0, nfp_score=0.0, file_hash_match=True)
    assert r["is_duplicate"] is True
    assert r["confidence"] == 1.0


def test_chromaprint_match_without_nfp_gets_penalty():
    r = combine_scores(chromaprint_score=0.90)
    assert r["is_duplicate"] is True
    assert r["confidence"] == pytest.approx(0.90 * 0.9)


def test_chromaprint_and_nfp_both_high():
    r = combine_scores(chromaprint_score=0.90, nfp_score=0.96)
    assert r["is_duplicate"] is True
    assert r["confidence"] == pytest.approx((0.90 + 0.96) / 2)


def test_nfp_contradicts_chromaprint_is_rejected():
    # This is the anti-false-positive guard.
    r = combine_scores(chromaprint_score=0.99, nfp_score=0.50)
    assert r["is_duplicate"] is False
    assert r["confidence"] == 0.0


def test_below_chromaprint_threshold_is_not_duplicate():
    r = combine_scores(chromaprint_score=0.50)
    assert r["is_duplicate"] is False
    assert r["confidence"] == 0.0


def test_exactly_at_thresholds_counts_as_match():
    r = combine_scores(chromaprint_score=0.85, nfp_score=0.90)
    assert r["is_duplicate"] is True
    assert r["confidence"] == pytest.approx((0.85 + 0.90) / 2)


def test_custom_thresholds_are_respected():
    # With a stricter Chromaprint threshold, 0.85 no longer qualifies.
    r = combine_scores(chromaprint_score=0.85, chromaprint_threshold=0.95)
    assert r["is_duplicate"] is False

    # With a looser NFP threshold, 0.70 confirms the match.
    r = combine_scores(chromaprint_score=0.90, nfp_score=0.70, nfp_threshold=0.65)
    assert r["is_duplicate"] is True


def test_nfp_metadata_passed_through():
    r = combine_scores(
        chromaprint_score=0.90,
        nfp_score=0.95,
        nfp_segments_matched=7,
        nfp_coverage=0.83,
    )
    assert r["nfp_segments_matched"] == 7
    assert r["nfp_coverage"] == 0.83
