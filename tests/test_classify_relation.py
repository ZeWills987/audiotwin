"""Unit tests for classify_relation — pure decision logic, no audio decoding."""

import pytest

from audiotwin import classify_relation


def test_high_chromaprint_is_duplicate_regardless_of_nfp():
    assert classify_relation(0.95, nfp_score=None)["relation_type"] == "DUPLICATE"
    assert classify_relation(0.95, nfp_score=0.10)["relation_type"] == "DUPLICATE"
    assert classify_relation(0.95, nfp_score=0.99)["relation_type"] == "DUPLICATE"


def test_mid_chromaprint_with_high_nfp_is_remaster():
    r = classify_relation(0.70, nfp_score=0.95)
    assert r["relation_type"] == "REMASTER"
    assert r["confidence"] == pytest.approx(0.95 * 0.9)
    assert r["score_gap"] == pytest.approx(0.95 - 0.70)


def test_mid_chromaprint_with_low_nfp_is_no_relation():
    r = classify_relation(0.70, nfp_score=0.50)
    assert r["relation_type"] == "NO_RELATION"
    assert r["confidence"] == 0.0


def test_mid_chromaprint_without_nfp_is_no_relation():
    r = classify_relation(0.70, nfp_score=None)
    assert r["relation_type"] == "NO_RELATION"
    assert r["confidence"] == 0.0
    assert r["score_gap"] is None


def test_low_chromaprint_is_no_relation_even_with_strong_nfp():
    # Below remaster_chromaprint_min, the spectral link is too thin to trust
    # even if NFP suggests strong structural similarity.
    r = classify_relation(0.40, nfp_score=0.99)
    assert r["relation_type"] == "NO_RELATION"
    assert r["confidence"] == 0.0


def test_score_gap_computation():
    r = classify_relation(0.70, nfp_score=0.90)
    assert r["score_gap"] == pytest.approx(0.20)

    r_none = classify_relation(0.70, nfp_score=None)
    assert r_none["score_gap"] is None


def test_default_thresholds():
    # Right at the duplicate boundary -> DUPLICATE.
    assert classify_relation(0.85, nfp_score=None)["relation_type"] == "DUPLICATE"
    # Right at the remaster floor -> eligible for REMASTER given strong NFP.
    r = classify_relation(0.60, nfp_score=0.90)
    assert r["relation_type"] == "REMASTER"
    # Just below the remaster floor -> NO_RELATION even with strong NFP.
    r = classify_relation(0.599, nfp_score=0.99)
    assert r["relation_type"] == "NO_RELATION"


def test_custom_thresholds_are_respected():
    # Loosen the duplicate threshold so 0.80 now counts as DUPLICATE.
    r = classify_relation(0.80, nfp_score=None, duplicate_threshold=0.75)
    assert r["relation_type"] == "DUPLICATE"

    # Tighten remaster_chromaprint_min so 0.65 no longer qualifies for REMASTER.
    r = classify_relation(0.65, nfp_score=0.95, remaster_chromaprint_min=0.70)
    assert r["relation_type"] == "NO_RELATION"

    # Loosen remaster_nfp_threshold so a weaker NFP score still confirms REMASTER.
    r = classify_relation(0.70, nfp_score=0.70, remaster_nfp_threshold=0.65)
    assert r["relation_type"] == "REMASTER"

    # Widen remaster_chromaprint_max above the duplicate threshold.
    r = classify_relation(
        0.82, nfp_score=0.95, duplicate_threshold=0.90, remaster_chromaprint_max=0.90
    )
    assert r["relation_type"] == "REMASTER"


def test_duplicate_confidence_matches_combine_scores_logic():
    # Without NFP: chromaprint_score * 0.9 (same penalty as combine_scores).
    r = classify_relation(0.90, nfp_score=None)
    assert r["confidence"] == pytest.approx(0.90 * 0.9)

    # With confirming NFP: average of the two scores.
    r = classify_relation(0.90, nfp_score=0.95)
    assert r["confidence"] == pytest.approx((0.90 + 0.95) / 2)
