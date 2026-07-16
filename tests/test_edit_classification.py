"""Tests for EDIT classification — pure geometry on match points, no audio."""

import numpy as np
import pytest

from audiotwin import classify_edit, compute_coverage, fit_temporal_alignment

SEED = 42


def _line_matches(slope, intercept, t_query_points, score=1.0):
    return [(float(t), float(slope * t + intercept), score) for t in t_query_points]


def test_full_match_slope_one_full_coverage():
    # Points every 5s across a 180s query mapping 1:1 onto a 180s ref.
    matches = _line_matches(1.0, 0.0, np.arange(0, 181, 5))
    r = classify_edit(matches, query_duration=180, ref_duration=180, random_seed=SEED)
    assert r["edit_type_hint"] == "full_match"
    assert r["slope"] == pytest.approx(1.0, abs=0.01)
    assert r["confidence"] > 0.9


def test_trim_detected_when_ref_coverage_low():
    # Radio edit: the query (90s edit) maps onto only the first half of a
    # 200s reference -> ref coverage well below the full threshold.
    matches = _line_matches(1.0, 0.0, np.arange(0, 91, 5))
    r = classify_edit(matches, query_duration=90, ref_duration=200, random_seed=SEED)
    assert r["edit_type_hint"] == "trim_or_extend"
    assert r["coverage_ref"] < 0.5


def test_sped_up_detected():
    # Sped-up ~30%: 1s of query consumes 1.3s of reference.
    matches = _line_matches(1.3, 0.0, np.arange(0, 139, 4))
    r = classify_edit(matches, query_duration=138, ref_duration=180, random_seed=SEED)
    assert r["edit_type_hint"] == "speed_change"
    assert r["slope"] == pytest.approx(1.3, abs=0.02)


def test_slowed_down_detected():
    matches = _line_matches(0.8, 0.0, np.arange(0, 226, 5))
    r = classify_edit(matches, query_duration=225, ref_duration=180, random_seed=SEED)
    assert r["edit_type_hint"] == "speed_change"
    assert r["slope"] == pytest.approx(0.8, abs=0.02)


def test_too_few_points_fails_fast_without_ransac(monkeypatch):
    # With fewer than min_inliers points, fit_temporal_alignment must bail
    # before sampling anything — patch the RNG factory to prove RANSAC never
    # actually starts.
    calls = []
    original_rng = np.random.default_rng
    monkeypatch.setattr(
        np.random, "default_rng", lambda *a, **kw: calls.append(1) or original_rng(*a, **kw)
    )

    matches = _line_matches(1.0, 0.0, [0, 10, 20, 30, 40])  # 5 points < 6
    fit = fit_temporal_alignment(matches)
    assert fit["fit_succeeded"] is False
    assert fit["inlier_indices"] == []
    assert calls == []

    r = classify_edit(matches, query_duration=180, ref_duration=180)
    assert r["edit_type_hint"] == "no_relation"
    assert r["confidence"] == 0.0


def test_random_scatter_yields_no_relation():
    # Simulates upstream matching false positives: no consistent line exists.
    rng = np.random.default_rng(SEED)
    matches = [(float(t), float(rng.uniform(0, 300)), 0.5) for t in rng.uniform(0, 180, size=30)]
    r = classify_edit(matches, query_duration=180, ref_duration=300, random_seed=SEED)
    assert r["edit_type_hint"] == "no_relation"
    assert r["confidence"] == 0.0


def test_outliers_are_rejected_but_line_recovered():
    # A clean slope-1 line plus scattered junk: RANSAC should recover the
    # line and flag the junk as outliers.
    clean = _line_matches(1.0, 2.0, np.arange(0, 181, 5))
    rng = np.random.default_rng(SEED)
    junk = [(float(rng.uniform(0, 180)), float(rng.uniform(200, 300)), 0.3) for _ in range(8)]
    r = classify_edit(clean + junk, query_duration=180, ref_duration=185, random_seed=SEED)
    assert r["edit_type_hint"] == "full_match"
    assert r["outlier_count"] == 8
    assert r["inlier_count"] == len(clean)


def test_compute_coverage_grouped_vs_scattered():
    matches = _line_matches(1.0, 0.0, [0, 5, 10, 15, 20, 25])
    cov = compute_coverage(matches, list(range(len(matches))), 100, 100)
    assert cov["coverage_query"] == pytest.approx(0.25)
    assert cov["coverage_ref"] == pytest.approx(0.25)
    assert cov["is_consecutive"] is True

    # Two clusters separated by a 60s hole -> not consecutive.
    gappy = _line_matches(1.0, 0.0, [0, 5, 10, 70, 75, 80])
    cov = compute_coverage(gappy, list(range(len(gappy))), 100, 100)
    assert cov["is_consecutive"] is False


def test_compute_coverage_empty_inliers():
    matches = _line_matches(1.0, 0.0, [0, 10, 20])
    cov = compute_coverage(matches, [], 100, 100)
    assert cov == {"coverage_query": 0.0, "coverage_ref": 0.0, "is_consecutive": False}


def test_slope_outside_bounds_rejected():
    # A perfect line at slope 3.0 is outside the default (0.5, 2.0) bounds.
    matches = _line_matches(3.0, 0.0, np.arange(0, 61, 5))
    fit = fit_temporal_alignment(matches, random_seed=SEED)
    assert fit["fit_succeeded"] is False

    # Widening the bounds makes the same data fit.
    fit = fit_temporal_alignment(matches, slope_bounds=(0.2, 4.0), random_seed=SEED)
    assert fit["fit_succeeded"] is True
    assert fit["slope"] == pytest.approx(3.0, abs=0.02)


def test_custom_thresholds():
    # min_inliers custom: 5 points pass with min_inliers=4.
    matches = _line_matches(1.0, 0.0, [0, 10, 20, 30, 40])
    fit = fit_temporal_alignment(matches, min_inliers=4, random_seed=SEED)
    assert fit["fit_succeeded"] is True

    # speed_change_epsilon custom: slope 1.02 is full_match by default
    # (epsilon 0.03) but speed_change with a tighter epsilon.
    near_one = _line_matches(1.02, 0.0, np.arange(0, 181, 5))
    r_default = classify_edit(near_one, query_duration=180, ref_duration=185, random_seed=SEED)
    assert r_default["edit_type_hint"] == "full_match"
    r_tight = classify_edit(
        near_one,
        query_duration=180,
        ref_duration=185,
        speed_change_epsilon=0.01,
        random_seed=SEED,
    )
    assert r_tight["edit_type_hint"] == "speed_change"

    # full_coverage_threshold custom: 80% coverage is trim_or_extend by
    # default (threshold 0.90) but full_match with a looser threshold.
    partial = _line_matches(1.0, 0.0, np.arange(0, 145, 5))  # spans 140s of 180s
    r_default = classify_edit(partial, query_duration=180, ref_duration=180, random_seed=SEED)
    assert r_default["edit_type_hint"] == "trim_or_extend"
    r_loose = classify_edit(
        partial,
        query_duration=180,
        ref_duration=180,
        full_coverage_threshold=0.75,
        random_seed=SEED,
    )
    assert r_loose["edit_type_hint"] == "full_match"


def test_confidence_favors_absolute_evidence():
    # 20 inliers / 25 points should beat 6 inliers / 6 points: more absolute
    # evidence outweighs a perfect ratio on a tiny sample.
    rng = np.random.default_rng(SEED)
    many = _line_matches(1.0, 0.0, np.arange(0, 100, 5))  # 20 clean points
    junk = [(float(rng.uniform(0, 100)), float(rng.uniform(300, 400)), 0.2) for _ in range(5)]
    r_many = classify_edit(many + junk, query_duration=100, ref_duration=100, random_seed=SEED)

    few = _line_matches(1.0, 0.0, [0, 20, 40, 60, 80, 100])  # exactly 6 points
    r_few = classify_edit(few, query_duration=100, ref_duration=100, random_seed=SEED)

    assert r_many["inlier_count"] == 20
    assert r_few["inlier_count"] == 6
    assert r_many["confidence"] >= r_few["confidence"]


def test_intercept_recovered():
    # Query starts 30s into the reference.
    matches = _line_matches(1.0, 30.0, np.arange(0, 121, 5))
    fit = fit_temporal_alignment(matches, random_seed=SEED)
    assert fit["fit_succeeded"] is True
    assert fit["intercept"] == pytest.approx(30.0, abs=0.1)


def test_score_weighted_sampling_prefers_trusted_points():
    # Two competing lines: the true one carries high match scores, a decoy
    # of equal size carries near-zero scores. Weighted sampling must seed
    # from the trusted points and recover the true line.
    trusted = _line_matches(1.0, 0.0, np.arange(0, 46, 5), score=1.0)  # 10 points
    decoy = _line_matches(1.5, 90.0, np.arange(0, 46, 5), score=1e-6)  # 10 points
    fit = fit_temporal_alignment(trusted + decoy, random_seed=SEED)
    assert fit["fit_succeeded"] is True
    assert fit["slope"] == pytest.approx(1.0, abs=0.02)
    assert fit["intercept"] == pytest.approx(0.0, abs=0.5)

    # Uniform sampling on the same data is a coin flip between the two
    # lines (both have 10 inliers) — only assert it still fits SOME line.
    fit_uniform = fit_temporal_alignment(trusted + decoy, random_seed=SEED, weight_by_score=False)
    assert fit_uniform["fit_succeeded"] is True


def test_all_zero_scores_fall_back_to_uniform_sampling():
    matches = _line_matches(1.0, 0.0, np.arange(0, 61, 5), score=0.0)
    fit = fit_temporal_alignment(matches, random_seed=SEED)
    assert fit["fit_succeeded"] is True
    assert fit["slope"] == pytest.approx(1.0, abs=0.02)


def test_adaptive_termination_stops_early_on_clean_data(monkeypatch):
    # On all-inlier data the Fischler-Bolles bound is tiny, so sampling
    # must stop well before the 1000-iteration budget. Count actual draws.
    draws = []
    original_rng = np.random.default_rng

    class CountingRng:
        def __init__(self, *args, **kwargs):
            self._rng = original_rng(*args, **kwargs)

        def choice(self, *a, **kw):
            draws.append(1)
            return self._rng.choice(*a, **kw)

        def __getattr__(self, name):
            return getattr(self._rng, name)

    monkeypatch.setattr(np.random, "default_rng", CountingRng)

    matches = _line_matches(1.0, 0.0, np.arange(0, 101, 5))
    fit = fit_temporal_alignment(matches, random_seed=SEED)
    assert fit["fit_succeeded"] is True
    assert len(draws) < 100  # far below the 1000-iteration budget
