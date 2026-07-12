"""Tests for the landmark module (Wang 2003 fingerprinting + SAMPLE/MASHUP).

Fixtures are synthetic tone sequences (numpy + soundfile, generated on the
fly). Landmark matching needs time-varying spectral content, so fixtures are
random-but-seeded sequences of short chords rather than stationary sines.

These tests need ffmpeg (for decode_audio) and scipy (the [landmark] extra);
they are skipped when either is unavailable.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest
import soundfile as sf

scipy = pytest.importorskip("scipy", reason="landmark extra (scipy) not installed")

from audiotwin.landmark import (  # noqa: E402
    LandmarkIndex,
    classify_mashup,
    classify_sample,
    extract_landmarks,
)

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg not installed"
)

SR = 11025


def _tone_sequence(seconds: float, seed: int, sr: int = SR) -> np.ndarray:
    """A random-but-seeded sequence of 250 ms chords — rich, time-varying
    spectral content that produces plenty of distinctive landmarks."""
    rng = np.random.default_rng(seed)
    n_steps = int(seconds * 4)
    chunks = []
    t = np.linspace(0.0, 0.25, int(0.25 * sr), endpoint=False)
    for _ in range(n_steps):
        freqs = rng.uniform(200, 4000, size=3)
        chord = sum(np.sin(2 * np.pi * f * t) for f in freqs)
        envelope = np.minimum(1.0, 20 * np.minimum(t, 0.25 - t))  # declick
        chunks.append(chord * envelope)
    signal = np.concatenate(chunks)
    return (0.5 * signal / np.max(np.abs(signal))).astype(np.float32)


@pytest.fixture(scope="module")
def track_a():
    return _tone_sequence(60.0, seed=1)


@pytest.fixture(scope="module")
def track_b():
    return _tone_sequence(30.0, seed=2)


@pytest.fixture(scope="module")
def track_c():
    return _tone_sequence(30.0, seed=3)


def _write(tmp_path, name, audio, sr=SR):
    path = str(tmp_path / name)
    sf.write(path, audio, sr)
    return path


@requires_ffmpeg
def test_localized_sample_detected(tmp_path, track_a):
    index = LandmarkIndex(":memory:")
    index.add_track("A", _write(tmp_path, "a.wav", track_a))

    # Query: 10 s extract of A (from t=20 s in A) at the START of the query,
    # followed by 20 s of unrelated background -> offset = 20 - 0 = 20 s.
    extract = track_a[20 * SR : 30 * SR]
    background = _tone_sequence(20.0, seed=99)
    query = np.concatenate([extract, background])
    query_path = _write(tmp_path, "query.wav", query)

    results = index.query(query_path)
    assert results, "expected the sample's source track to be found"
    top = results[0]
    assert top["track_id"] == "A"
    assert top["offset_seconds"] == pytest.approx(20.0, abs=0.5)
    assert top["match_points"]

    sample = classify_sample(top, query_duration=30.0, ref_duration=60.0)
    assert sample["is_localized_match"] is True
    # End bounds are anchor times, biased early by up to the target zone's
    # dt_max (3 s): anchors in the last seconds of the sample can't pair
    # inside the fragment, so the last *matching* anchor sits ~dt_max before
    # the true fragment end. Start bounds don't have this bias.
    assert sample["sample_start_query"] == pytest.approx(0.0, abs=1.5)
    assert sample["sample_end_query"] == pytest.approx(10.0, abs=3.5)
    assert sample["sample_start_ref"] == pytest.approx(20.0, abs=1.5)
    assert sample["sample_end_ref"] == pytest.approx(30.0, abs=3.5)
    assert sample["confidence"] > 0.0


@requires_ffmpeg
def test_full_track_reencoded_matches_globally(tmp_path, track_a):
    import subprocess

    index = LandmarkIndex(":memory:")
    index.add_track("A", _write(tmp_path, "a.wav", track_a))

    # mp3: libmp3lame natively supports the 11025 Hz fixture rate (vorbis
    # rejects some low mono rates at fixed bitrates).
    reencoded = str(tmp_path / "a_reenc.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_path / "a.wav"), "-b:a", "64k", reencoded],
        check=True,
        capture_output=True,
    )

    results = index.query(reencoded)
    assert results and results[0]["track_id"] == "A"
    top = results[0]
    assert top["offset_seconds"] == pytest.approx(0.0, abs=0.5)

    sample = classify_sample(top, query_duration=60.0, ref_duration=60.0)
    # Global relation, not a localized sample.
    assert sample["coverage_query"] > 0.8
    assert sample["is_localized_match"] is False
    assert sample["confidence"] == 0.0


@requires_ffmpeg
def test_unrelated_noise_has_no_match(tmp_path, track_a):
    index = LandmarkIndex(":memory:")
    index.add_track("A", _write(tmp_path, "a.wav", track_a))

    rng = np.random.default_rng(123)
    noise = (0.3 * rng.standard_normal(20 * SR)).astype(np.float32)
    results = index.query(_write(tmp_path, "noise.wav", noise))
    assert results == []


@requires_ffmpeg
def test_mashup_of_three_sources(tmp_path, track_a, track_b, track_c):
    index = LandmarkIndex(":memory:")
    index.add_track("A", _write(tmp_path, "a.wav", track_a))
    index.add_track("B", _write(tmp_path, "b.wav", track_b))
    index.add_track("C", _write(tmp_path, "c.wav", track_c))

    # 30 s mashup: 10 s from each indexed track, concatenated.
    mashup = np.concatenate([track_a[: 10 * SR], track_b[: 10 * SR], track_c[: 10 * SR]])
    results = index.query(_write(tmp_path, "mashup.wav", mashup))

    verdict = classify_mashup(results, query_duration=30.0)
    assert verdict["is_mashup_pattern"] is True
    assert verdict["source_count"] == 3
    assert verdict["coverage_total"] > 0.7

    by_track = {s["track_id"]: s for s in verdict["sources"]}
    assert by_track["A"]["region_start"] == pytest.approx(0.0, abs=1.5)
    assert by_track["B"]["region_start"] == pytest.approx(10.0, abs=1.5)
    assert by_track["C"]["region_start"] == pytest.approx(20.0, abs=1.5)
    assert verdict["confidence"] > 0.0


@requires_ffmpeg
def test_single_source_is_not_mashup(tmp_path, track_a):
    index = LandmarkIndex(":memory:")
    index.add_track("A", _write(tmp_path, "a.wav", track_a))
    results = index.query(_write(tmp_path, "q.wav", track_a[: 20 * SR]))
    verdict = classify_mashup(results, query_duration=20.0)
    assert verdict["is_mashup_pattern"] is False
    assert verdict["confidence"] == 0.0


@requires_ffmpeg
def test_pitch_shifted_query_found_with_pitch_range(tmp_path, track_a):
    librosa = pytest.importorskip("librosa", reason="cover extra (librosa) not installed")

    index = LandmarkIndex(":memory:")
    index.add_track("A", _write(tmp_path, "a.wav", track_a))

    shifted = librosa.effects.pitch_shift(track_a[: 20 * SR], sr=SR, n_steps=1)
    query_path = _write(tmp_path, "shifted.wav", shifted.astype(np.float32))

    # Without pitch compensation the shifted query should match poorly...
    plain = index.query(query_path)
    plain_hashes = plain[0]["aligned_hashes"] if plain else 0

    # ...but with pitch_shift_range=2 the -1 semitone variant recovers it.
    compensated = index.query(query_path, pitch_shift_range=2)
    assert compensated and compensated[0]["track_id"] == "A"
    assert compensated[0]["aligned_hashes"] > plain_hashes
    assert compensated[0]["pitch_shift_semitones"] != 0


@requires_ffmpeg
def test_add_existing_track_id_raises(tmp_path, track_b):
    index = LandmarkIndex(":memory:")
    path = _write(tmp_path, "b.wav", track_b)
    index.add_track("B", path)
    with pytest.raises(ValueError, match="already indexed"):
        index.add_track("B", path)


@requires_ffmpeg
def test_remove_track(tmp_path, track_b):
    index = LandmarkIndex(":memory:")
    path = _write(tmp_path, "b.wav", track_b)
    index.add_track("B", path)
    assert index.track_ids() == ["B"]

    removed = index.remove_track("B")
    assert removed > 0
    assert index.track_ids() == []
    assert index.query(path) == []


@pytest.mark.slow
def test_extract_landmarks_performance():
    audio = _tone_sequence(180.0, seed=7)
    start = time.perf_counter()
    landmarks = extract_landmarks(audio)
    elapsed = time.perf_counter() - start
    assert landmarks, "expected landmarks from 3 minutes of tonal content"
    assert elapsed < 5.0, f"extract_landmarks took {elapsed:.2f}s on 3 min of audio"
