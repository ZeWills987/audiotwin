"""Tests for the neural module (Sony Sample-ID bridge).

These need PyTorch AND the sampleid package (installed from GitHub), plus
ffmpeg; the first run downloads the official checkpoint from Zenodo. They
are skipped automatically when any of those is unavailable.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch", reason="neural extra (torch) not installed")
sampleid = pytest.importorskip("sampleid", reason="sampleid package not installed")

from audiotwin.neural import (  # noqa: E402
    NEURAL_SR,
    neural_embedding,
    neural_match_points,
    neural_similarity,
)

requires_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _rich_audio(seconds: float, seed: int) -> np.ndarray:
    """Noise + dense tone mix — spectrally rich like real music."""
    rng = np.random.default_rng(seed)
    n = int(seconds * NEURAL_SR)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    sig = 0.3 * rng.standard_normal(n)
    for f in rng.uniform(100, 6000, 20):
        sig += 0.1 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return (0.5 * sig / np.max(np.abs(sig))).astype(np.float32)


@pytest.fixture(scope="module")
def track_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("neural")
    a = _rich_audio(30.0, seed=1)
    b = _rich_audio(30.0, seed=2)
    paths = {}
    for name, audio in [("a", a), ("b", b)]:
        p = str(d / f"{name}.wav")
        sf.write(p, audio, NEURAL_SR)
        paths[name] = p
    # A remastered-ish variant of a: mild lowpass blend + gain change.
    kernel = np.ones(9) / 9
    remaster = 0.6 * a + 0.4 * np.convolve(a, kernel, mode="same")
    remaster = (0.9 * remaster / np.max(np.abs(remaster))).astype(np.float32)
    p = str(d / "a_remaster.wav")
    sf.write(p, remaster, NEURAL_SR)
    paths["a_remaster"] = p
    return paths


@requires_ffmpeg
def test_embedding_shape_and_normalization(track_paths):
    emb = neural_embedding(track_paths["a"])
    # 30 s at 5 s chunks / 2.5 s hop -> 11 chunks.
    assert emb.shape[0] == 11
    assert emb.shape[1] > 0
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


@requires_ffmpeg
def test_self_similarity_is_high_and_unrelated_lower(track_paths):
    self_sim = neural_similarity(track_paths["a"], track_paths["a"])
    other_sim = neural_similarity(track_paths["a"], track_paths["b"])

    assert self_sim["nfp_score"] == pytest.approx(1.0, abs=1e-4)
    assert self_sim["nfp_coverage"] == 1.0
    assert other_sim["nfp_score"] < self_sim["nfp_score"]


@requires_ffmpeg
def test_remaster_variant_scores_above_unrelated(track_paths):
    remaster = neural_similarity(track_paths["a"], track_paths["a_remaster"])
    unrelated = neural_similarity(track_paths["a"], track_paths["b"])
    # The neural signal must recognize the remastered variant as far more
    # similar than unrelated content — this is exactly the REMASTER
    # signature detect_relation() needs.
    assert remaster["nfp_score"] > unrelated["nfp_score"]


def test_localized_match_geometry_gate():
    """Unit-test the localized-match mechanism on CRAFTED embeddings.

    Synthetic AUDIO cannot exercise this path: the model (trained on real
    music) embeds all out-of-domain tone/noise fixtures as near-identical,
    saturating the background. The model's discrimination on real audio is
    validated separately (a real overdubbed sample was found at the exact
    offset while whole-track nfp was 0.08); here we verify the geometry:
    a coherent diagonal of matching chunks is found and fitted, scattered
    junk is rejected by the RANSAC gate.
    """
    from audiotwin.neural import _localized_match_from_embeddings

    rng = np.random.default_rng(42)

    def unit_rows(n, dim=64):
        m = rng.standard_normal((n, dim)).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    emb_a = unit_rows(24)  # 60 s query at 2.5 s hop
    emb_b = unit_rows(12)  # 30 s reference
    # Fragment: query chunks 8..13 ARE reference chunks 2..7 (raw cos 1.0)
    # -> t_query 22.5 s maps to t_ref 7.5 s: slope 1, offset -15 s.
    emb_a[8:14] = emb_b[2:8]

    kwargs = dict(
        match_threshold=0.25,
        min_inliers=4,
        residual_threshold=2.6,
        chunk_seconds=5.0,
        chunk_hop_seconds=2.5,
        cosine_floor=0.95,
    )
    loc = _localized_match_from_embeddings(emb_a, emb_b, **kwargs)
    assert loc["found"] is True
    assert loc["slope"] == pytest.approx(1.0, abs=0.05)
    assert loc["offset_seconds"] == pytest.approx(-15.0, abs=1.5)
    assert loc["inlier_count"] >= 6
    assert loc["match_start_query"] == pytest.approx(20.0, abs=2.6)
    assert loc["match_start_ref"] == pytest.approx(5.0, abs=2.6)
    # Localized: the fragment covers a minority of the query.
    assert loc["coverage_query"] < 0.35

    # Random embeddings (raw cosines ~0 << floor): nothing to find.
    none = _localized_match_from_embeddings(unit_rows(24), unit_rows(12), **kwargs)
    assert none["found"] is False


@requires_ffmpeg
def test_match_points_feed_classify_edit(track_paths):
    from audiotwin import classify_edit

    points = neural_match_points(track_paths["a"], track_paths["a"])
    assert points, "self-match must produce match points"
    # Identity alignment: slope 1, intercept 0. Points sit at chunk CENTERS
    # (2.5 s..27.5 s on a 30 s track), so max coverage is 25/30 ≈ 0.83 —
    # lower the full-coverage threshold accordingly (see the docstring).
    verdict = classify_edit(
        points,
        query_duration=30.0,
        ref_duration=30.0,
        full_coverage_threshold=0.80,
        random_seed=42,
    )
    assert verdict["edit_type_hint"] == "full_match"
    assert verdict["slope"] == pytest.approx(1.0, abs=0.05)
