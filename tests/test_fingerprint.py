"""Tests for the level-1 Chromaprint fingerprint + comparison.

These require the ``fpcalc`` binary to be installed and are skipped
automatically when it is unavailable.
"""

import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from audiotwin import (
    AudioTooShortError,
    compare_fingerprints,
    compute_fingerprint,
    detect,
)
from audiotwin.core import FPCALC_COMMAND, FPCALC_COMMAND_ENVVAR, _decode

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


# --- uint32 overflow regression (real-world fingerprints) --------------------
#
# Chromaprint sub-fingerprints are UNSIGNED 32-bit words. Spectrally rich
# audio (noise, dense mixes — i.e. real music) routinely produces words
# above 2**31 - 1, which overflowed the previous np.int32 handling with
# "OverflowError: Python integer ... out of bounds for int32". The original
# pure-sine fixtures happened to only produce small words, which is why the
# test suite (and CI) never caught it — hence the rich fixtures below.

SR = 44100


def _rich_audio(seconds: float, seed: int) -> np.ndarray:
    """Noise + dense tone mix: spectrally rich like real music, guaranteed
    (asserted in the tests) to produce fingerprint words above int32 max."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    sig = 0.3 * rng.standard_normal(n)
    for f in rng.uniform(100, 8000, 20):
        sig += 0.1 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return (0.5 * sig / np.max(np.abs(sig))).astype(np.float32)


@requires_chromaprint
def test_fingerprint_words_above_int32_max_are_handled(tmp_path):
    path = str(tmp_path / "rich.wav")
    sf.write(path, _rich_audio(30.0, seed=42), SR)

    fp = compute_fingerprint(path)  # crashed with OverflowError before the fix

    words = _decode(fp)
    assert words.dtype == np.uint32
    # Self-validating: the fixture MUST exercise the >2**31-1 range, else
    # this regression test silently stops testing anything.
    assert int(words.max()) > 2**31 - 1, (
        "fixture no longer produces large fingerprint words — regression "
        "test would be vacuous; change the seed/content"
    )

    assert compare_fingerprints(fp, fp) == pytest.approx(1.0)


@requires_chromaprint
def test_detect_end_to_end_on_minutes_long_rich_audio(tmp_path):
    """Full detect() on two multi-minute, spectrally rich files: the same
    content re-encoded (duplicate) and unrelated content (distinct)."""
    import subprocess

    original = str(tmp_path / "long_a.wav")
    sf.write(original, _rich_audio(150.0, seed=1), SR)

    reencoded = str(tmp_path / "long_a.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", original, "-b:a", "128k", reencoded],
        check=True,
        capture_output=True,
    )

    unrelated = str(tmp_path / "long_b.wav")
    sf.write(unrelated, _rich_audio(150.0, seed=2), SR)

    dup = detect(original, reencoded)
    assert dup["file_hash_match"] is False
    assert dup["chromaprint_score"] > 0.85
    assert dup["is_duplicate"] is True

    distinct = detect(original, unrelated)
    assert distinct["is_duplicate"] is False
