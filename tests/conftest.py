"""Shared test fixtures.

All audio fixtures are generated synthetically with numpy + soundfile — no
binary audio files are committed to the repo.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

SAMPLE_RATE = 44100


def _write(path, samples: np.ndarray, samplerate: int = SAMPLE_RATE) -> str:
    sf.write(path, samples.astype(np.float32), samplerate)
    return str(path)


def _sine(freq: float, seconds: float, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * samplerate), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _chord(freqs, seconds: float, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """A richer signal (sum of tones) so Chromaprint has spectral content."""
    sig = sum(_sine(f, seconds, samplerate) for f in freqs)
    return sig / np.max(np.abs(sig))


@pytest.fixture
def sine_440(tmp_path):
    """A 20s harmonic tone written as a WAV file."""
    return _write(tmp_path / "sine_440.wav", _chord([220, 440, 660, 880], 20.0))


@pytest.fixture
def sine_440_copy(tmp_path):
    """A byte-identical copy of ``sine_440`` (same samples, same path family)."""
    samples = _chord([220, 440, 660, 880], 20.0)
    return _write(tmp_path / "sine_440_copy.wav", samples)


@pytest.fixture
def different_audio(tmp_path):
    """A completely different 20s signal (unrelated frequencies)."""
    return _write(tmp_path / "different.wav", _chord([311, 523, 784, 987], 20.0))


@pytest.fixture
def too_short(tmp_path):
    """A 5s file — below the 10s fingerprint floor."""
    return _write(tmp_path / "too_short.wav", _sine(440, 5.0))
