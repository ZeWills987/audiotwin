"""Tests for the cover module (chroma + OTI + DTW).

Fixtures are synthetic chord progressions: chords as sums of sines, with
two different harmonic recipes standing in for two "instruments". Requires
librosa (the [cover] extra) and ffmpeg for the file-based entry point.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

librosa = pytest.importorskip("librosa", reason="cover extra (librosa) not installed")

from audiotwin.cover import (  # noqa: E402
    compute_chroma,
    cover_similarity,
    cover_similarity_from_chroma,
    optimal_transposition,
)

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg not installed"
)

SR = 22050

# A simple I–vi–IV–V progression in C major, as MIDI note numbers.
PROGRESSION = [
    (60, 64, 67),  # C
    (57, 60, 64),  # Am
    (53, 57, 60),  # F
    (55, 59, 62),  # G
]

OTHER_PROGRESSION = [
    (61, 66, 68),  # unrelated cluster
    (63, 68, 70),
    (58, 61, 66),
    (56, 63, 65),
]


def _midi_to_hz(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def _render(
    progression,
    chord_seconds: float,
    harmonics: tuple[float, ...],
    transpose_semitones: int = 0,
    repeats: int = 4,
    sr: int = SR,
) -> np.ndarray:
    """Render a chord progression with a given harmonic recipe
    ("instrument"). Different harmonic weights = different timbre, same
    composition."""
    t = np.linspace(0.0, chord_seconds, int(chord_seconds * sr), endpoint=False)
    chunks = []
    for _ in range(repeats):
        for chord in progression:
            wave = np.zeros_like(t)
            for note in chord:
                f0 = _midi_to_hz(note + transpose_semitones)
                for k, weight in enumerate(harmonics, start=1):
                    wave += weight * np.sin(2 * np.pi * f0 * k * t)
            envelope = np.minimum(1.0, 30 * np.minimum(t, chord_seconds - t))
            chunks.append(wave * envelope)
    signal = np.concatenate(chunks)
    return (0.5 * signal / np.max(np.abs(signal))).astype(np.float32)


INSTRUMENT_A = (1.0, 0.5, 0.25, 0.12)  # bright, many harmonics
INSTRUMENT_B = (1.0, 0.15)  # mellow, near-sinusoidal


@pytest.fixture(scope="module")
def original():
    return _render(PROGRESSION, 1.0, INSTRUMENT_A)


@pytest.fixture(scope="module")
def cover_transposed_stretched():
    # Same composition: different "instrument", +3 semitones, 15% slower.
    return _render(PROGRESSION, 1.15, INSTRUMENT_B, transpose_semitones=3)


@pytest.fixture(scope="module")
def unrelated():
    return _render(OTHER_PROGRESSION, 0.9, INSTRUMENT_B)


def test_cover_detected_with_transposition(original, cover_transposed_stretched, unrelated):
    chroma_orig = compute_chroma(original)
    chroma_cover = compute_chroma(cover_transposed_stretched)
    chroma_unrelated = compute_chroma(unrelated)

    cover = cover_similarity_from_chroma(chroma_orig, chroma_cover)
    other = cover_similarity_from_chroma(chroma_orig, chroma_unrelated)

    assert cover["transposition_semitones"] == 3
    # Relative ordering, not absolute values: the true cover must score
    # clearly above the unrelated progression.
    assert cover["similarity"] > other["similarity"]
    assert cover["similarity"] > 0.7


def test_optimal_transposition_recovers_shift(original, cover_transposed_stretched):
    chroma_a = compute_chroma(original)
    chroma_b = compute_chroma(cover_transposed_stretched)
    k, sim = optimal_transposition(chroma_a, chroma_b)
    assert k == 3
    assert sim > 0.5


@requires_ffmpeg
def test_from_files_equals_from_chroma(tmp_path, original, cover_transposed_stretched):
    path_a = str(tmp_path / "a.wav")
    path_b = str(tmp_path / "b.wav")
    sf.write(path_a, original, SR)
    sf.write(path_b, cover_transposed_stretched, SR)

    from audiotwin.audio import decode_audio

    file_result = cover_similarity(path_a, path_b)

    chroma_a = compute_chroma(decode_audio(path_a, sr=SR))
    chroma_b = compute_chroma(decode_audio(path_b, sr=SR))
    chroma_result = cover_similarity_from_chroma(chroma_a, chroma_b)

    assert file_result["similarity"] == pytest.approx(chroma_result["similarity"])
    assert (
        file_result["transposition_semitones"] == chroma_result["transposition_semitones"]
    )
    assert file_result["duration_ratio"] == pytest.approx(1.15, abs=0.02)


def test_no_hpss_path_works(original, cover_transposed_stretched):
    chroma_a = compute_chroma(original, use_hpss=False)
    chroma_b = compute_chroma(cover_transposed_stretched, use_hpss=False)
    result = cover_similarity_from_chroma(chroma_a, chroma_b)
    assert result["transposition_semitones"] == 3
    assert 0.0 <= result["similarity"] <= 1.0


def test_chroma_shape_and_normalization(original):
    chroma = compute_chroma(original, target_fps=2.0)
    assert chroma.shape[0] == 12
    norms = np.linalg.norm(chroma, axis=0)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-6)
    # ~2 fps over a 16 s render -> ~32 frames.
    expected_frames = 16 * 2
    assert abs(chroma.shape[1] - expected_frames) <= 3


# --- silence regression -------------------------------------------------------
#
# Audio containing silent passages produced all-zero chroma columns whose
# cosine distance is 0/0 = NaN, crashing librosa.sequence.dtw with
# "ParameterError: DTW cost matrix C has NaN values".


def _with_silence(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    """Wrap a signal with LONG total silence at the start, middle, and end.

    The gaps must be long (30 s): the CQT's low-frequency analysis windows
    span several seconds, so short gaps get bridged by leakage from the
    surrounding content and never yield zero chroma columns.
    """
    gap = np.zeros(30 * sr, dtype=np.float32)
    half = len(signal) // 2
    return np.concatenate([gap, signal[:half], gap, signal[half:], gap])


@requires_ffmpeg
def test_silent_passages_do_not_crash_cover_similarity(tmp_path, original, unrelated):
    path_a = str(tmp_path / "silent_a.wav")
    path_b = str(tmp_path / "silent_b.wav")
    sf.write(path_a, _with_silence(original), SR)
    sf.write(path_b, _with_silence(unrelated), SR)

    # Crashed with "ParameterError: DTW cost matrix C has NaN values" before
    # the fix. use_hpss=False keeps the test fast; the silence handling is
    # identical on both paths.
    result = cover_similarity(path_a, path_b, use_hpss=False)

    assert np.isfinite(result["similarity"])
    assert np.isfinite(result["dtw_normalized_cost"])
    assert 0.0 <= result["similarity"] <= 1.0


def test_silent_chroma_columns_are_zero_not_nan(original):
    chroma = compute_chroma(_with_silence(original), use_hpss=False)
    assert not np.isnan(chroma).any()
    # The silent gaps must actually produce zero columns, otherwise this
    # regression test stops exercising the silence path.
    norms = np.linalg.norm(chroma, axis=0)
    assert (norms < 1e-8).any(), "fixture produced no silent chroma frames"


def test_fully_silent_side_returns_finite_result(original):
    silent_chroma = compute_chroma(np.zeros(10 * SR, dtype=np.float32))
    chroma = compute_chroma(original)
    result = cover_similarity_from_chroma(chroma, silent_chroma)
    assert result["similarity"] == 0.0
    assert np.isfinite(result["dtw_normalized_cost"])
