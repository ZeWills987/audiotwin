"""Tests for classify_instrumental_pair — pure decision logic."""

import pytest

from audiotwin import classify_instrumental_pair


def test_vocal_a_instrumental_b():
    r = classify_instrumental_pair(0.85, vocal_coverage_a=0.70, vocal_coverage_b=0.05)
    assert r["is_instrumental_pair"] is True
    assert r["vocal_track"] == "a"
    assert r["instrumental_track"] == "b"
    assert r["vocal_gap"] == pytest.approx(0.65)
    assert r["confidence"] == pytest.approx(0.85 * 0.65)


def test_vocal_b_instrumental_a():
    r = classify_instrumental_pair(0.90, vocal_coverage_a=0.02, vocal_coverage_b=0.55)
    assert r["is_instrumental_pair"] is True
    assert r["vocal_track"] == "b"
    assert r["instrumental_track"] == "a"


def test_similar_content_but_both_vocal_is_not_pair():
    r = classify_instrumental_pair(0.90, vocal_coverage_a=0.60, vocal_coverage_b=0.70)
    assert r["is_instrumental_pair"] is False
    assert r["vocal_track"] is None
    assert r["instrumental_track"] is None
    assert r["confidence"] == 0.0


def test_similar_content_but_both_instrumental_is_not_pair():
    r = classify_instrumental_pair(0.90, vocal_coverage_a=0.05, vocal_coverage_b=0.03)
    assert r["is_instrumental_pair"] is False
    assert r["confidence"] == 0.0


def test_vocal_gap_but_different_content_is_not_pair():
    r = classify_instrumental_pair(0.30, vocal_coverage_a=0.80, vocal_coverage_b=0.02)
    assert r["is_instrumental_pair"] is False
    assert r["confidence"] == 0.0
    # Gap is still reported for diagnostics.
    assert r["vocal_gap"] == pytest.approx(0.78)


def test_ambiguous_middle_coverage_is_not_pair():
    # One side has vocals but the other is not clearly vocal-free (0.25 sits
    # between absent=0.10 and present=0.40).
    r = classify_instrumental_pair(0.90, vocal_coverage_a=0.70, vocal_coverage_b=0.25)
    assert r["is_instrumental_pair"] is False


def test_custom_thresholds():
    # Looser content threshold lets a weaker similarity qualify.
    r = classify_instrumental_pair(
        0.55, vocal_coverage_a=0.70, vocal_coverage_b=0.05, content_threshold=0.50
    )
    assert r["is_instrumental_pair"] is True

    # Looser absent threshold accepts a slightly leaky instrumental.
    r = classify_instrumental_pair(
        0.90, vocal_coverage_a=0.70, vocal_coverage_b=0.18, vocal_absent_threshold=0.20
    )
    assert r["is_instrumental_pair"] is True

    # Stricter present threshold rejects a mildly vocal track.
    r = classify_instrumental_pair(
        0.90, vocal_coverage_a=0.45, vocal_coverage_b=0.05, vocal_present_threshold=0.60
    )
    assert r["is_instrumental_pair"] is False


def test_exactly_at_thresholds():
    r = classify_instrumental_pair(0.70, vocal_coverage_a=0.40, vocal_coverage_b=0.10)
    assert r["is_instrumental_pair"] is True
