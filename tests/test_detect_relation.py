"""Tests for detect_relation — same spirit as test_fingerprint.py's detect() tests."""

import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from audiotwin import detect_relation
from audiotwin.core import FPCALC_COMMAND, FPCALC_COMMAND_ENVVAR

requires_chromaprint = pytest.mark.skipif(
    not shutil.which(os.environ.get(FPCALC_COMMAND_ENVVAR, FPCALC_COMMAND)),
    reason="Chromaprint (fpcalc) not installed",
)


@requires_chromaprint
def test_hash_match_short_circuits_without_computing_fingerprint(
    sine_440, sine_440_copy, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        "audiotwin.core.compute_fingerprint",
        lambda *a, **kw: calls.append((a, kw)) or "unused",
    )

    result = detect_relation(sine_440, sine_440_copy)

    assert calls == []
    assert result["file_hash_match"] is True
    assert result["relation_type"] == "DUPLICATE"
    assert result["confidence"] == 1.0
    assert result["chromaprint_score"] == 1.0


@requires_chromaprint
def test_skip_decode_if_hash_match_false_still_computes(sine_440, sine_440_copy, monkeypatch):
    import audiotwin.core as core

    calls = []
    original = core.compute_fingerprint
    monkeypatch.setattr(
        core, "compute_fingerprint", lambda *a, **kw: calls.append(1) or original(*a, **kw)
    )

    detect_relation(sine_440, sine_440_copy, skip_decode_if_hash_match=False)

    assert len(calls) == 2


@requires_chromaprint
def test_reencoded_audio_is_duplicate(tmp_path, sine_440):
    reencoded = str(tmp_path / "reencoded.ogg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", sine_440, "-b:a", "64k", reencoded],
        check=True,
        capture_output=True,
    )
    result = detect_relation(sine_440, reencoded)
    assert result["file_hash_match"] is False
    assert result["relation_type"] == "DUPLICATE"


@requires_chromaprint
def test_different_audio_is_no_relation(sine_440, different_audio):
    result = detect_relation(sine_440, different_audio)
    assert result["file_hash_match"] is False
    assert result["relation_type"] == "NO_RELATION"
    assert result["confidence"] == 0.0


@requires_chromaprint
def test_track_labels_are_passed_through(sine_440, different_audio):
    result = detect_relation(sine_440, different_audio)
    assert result["track_a"] == sine_440
    assert result["track_b"] == different_audio


@requires_chromaprint
def test_eq_gain_shift_reduces_chromaprint_similarity_more_than_reencoding(tmp_path, sine_440):
    """A synthetic "remaster" (EQ + gain change) should diverge more from the
    original, chromaprint-wise, than a plain bitrate re-encode of the exact
    same signal — a sanity check on relative direction, not an exact score.
    """
    from audiotwin import compare_fingerprints, compute_fingerprint

    data, sr = sf.read(sine_440)

    # Crude EQ: attenuate a moving-average (lowpass) blend and boost gain,
    # to shift spectral texture while keeping the same structural content.
    kernel = np.ones(9) / 9
    lowpassed = np.convolve(data, kernel, mode="same")
    remastered = 0.6 * data + 0.4 * lowpassed
    remastered = 1.4 * remastered / np.max(np.abs(remastered))
    remastered_path = str(tmp_path / "remastered.wav")
    sf.write(remastered_path, remastered.astype(np.float32), sr)

    reencoded_path = str(tmp_path / "reencoded.ogg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", sine_440, "-b:a", "64k", reencoded_path],
        check=True,
        capture_output=True,
    )

    base_fp = compute_fingerprint(sine_440)
    reencoded_score = compare_fingerprints(base_fp, compute_fingerprint(reencoded_path))
    remastered_score = compare_fingerprints(base_fp, compute_fingerprint(remastered_path))

    assert remastered_score <= reencoded_score


@requires_chromaprint
def test_synthetic_remaster_classified_via_detect_relation(sine_440, different_audio, monkeypatch):
    """With compare_fingerprints monkeypatched to a mid-range score (as a real
    EQ'd remaster might produce) and a caller-supplied high NFP score, the
    relation should be classified as REMASTER, not NO_RELATION or DUPLICATE.
    """
    monkeypatch.setattr("audiotwin.core.compare_fingerprints", lambda a, b: 0.70)

    result = detect_relation(sine_440, different_audio, nfp_score=0.95)

    assert result["relation_type"] == "REMASTER"
    assert result["score_gap"] == pytest.approx(0.95 - 0.70)
    assert result["confidence"] == pytest.approx(0.95 * 0.9)
