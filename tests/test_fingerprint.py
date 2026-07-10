"""Tests for the level-1 Chromaprint fingerprint + comparison.

These require the ``fpcalc`` binary to be installed and are skipped
automatically when it is unavailable.
"""

import os
import shutil

import pytest

from audiotwin import (
    AudioTooShortError,
    compare_fingerprints,
    compute_fingerprint,
    detect,
)
from audiotwin.core import FPCALC_COMMAND, FPCALC_COMMAND_ENVVAR

requires_chromaprint = pytest.mark.skipif(
    not shutil.which(os.environ.get(FPCALC_COMMAND_ENVVAR, FPCALC_COMMAND)),
    reason="Chromaprint (fpcalc) not installed",
)


@requires_chromaprint
def test_too_short_raises(too_short):
    with pytest.raises(AudioTooShortError):
        compute_fingerprint(too_short)


@requires_chromaprint
def test_identical_audio_fingerprints_match(sine_440, sine_440_copy):
    score = compare_fingerprints(
        compute_fingerprint(sine_440),
        compute_fingerprint(sine_440_copy),
    )
    assert score == pytest.approx(1.0, abs=1e-6)


@requires_chromaprint
def test_reencoded_audio_matches(tmp_path, sine_440):
    # Re-encode the same audio at a different bitrate → hash differs but the
    # Chromaprint fingerprint should still match strongly.
    import subprocess

    reencoded = str(tmp_path / "reencoded.ogg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", sine_440, "-b:a", "64k", reencoded],
        check=True,
        capture_output=True,
    )
    score = compare_fingerprints(
        compute_fingerprint(sine_440),
        compute_fingerprint(reencoded),
    )
    assert score >= 0.85


@requires_chromaprint
def test_different_audio_does_not_match(sine_440, different_audio):
    score = compare_fingerprints(
        compute_fingerprint(sine_440),
        compute_fingerprint(different_audio),
    )
    assert score < 0.85


@requires_chromaprint
def test_detect_identical_files_uses_hash(sine_440, sine_440_copy):
    result = detect(sine_440, sine_440_copy)
    assert result["file_hash_match"] is True
    assert result["is_duplicate"] is True
    assert result["confidence"] == 1.0


@requires_chromaprint
def test_detect_different_audio(sine_440, different_audio):
    result = detect(sine_440, different_audio)
    assert result["file_hash_match"] is False
    assert result["is_duplicate"] is False
    assert result["track_a"] == sine_440
    assert result["track_b"] == different_audio
