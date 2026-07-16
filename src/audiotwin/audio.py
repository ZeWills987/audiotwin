"""Shared in-memory audio decoding.

Every audiotwin module that needs PCM samples goes through
:func:`decode_audio`, which streams ffmpeg's raw float32 output straight
into a numpy buffer — no converted file is ever written to disk.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

#: Name of (or path to) the ffmpeg executable. Override via the
#: ``AUDIOTWIN_FFMPEG`` environment variable if it isn't on PATH.
FFMPEG_COMMAND_ENVVAR = "AUDIOTWIN_FFMPEG"
FFMPEG_COMMAND = "ffmpeg"

#: Safety-net timeout (seconds) for the ffmpeg subprocess — a corrupt file
#: or wedged decoder fails loudly instead of hanging the caller forever.
FFMPEG_TIMEOUT_SECONDS = 600.0


class FfmpegNotFoundError(RuntimeError):
    """Raised when the ``ffmpeg`` executable can't be found."""


def _ffmpeg_path() -> str:
    return os.environ.get(FFMPEG_COMMAND_ENVVAR, FFMPEG_COMMAND)


def decode_audio(path: str, sr: int, mono: bool = True) -> np.ndarray:
    """Decode an audio file to float32 PCM entirely in memory.

    Args:
        path: Path to the audio file (any format ffmpeg can read).
        sr: Target sample rate; ffmpeg resamples during decoding.
        mono: Downmix to a single channel (default True). When False, the
            returned array is interleaved as ffmpeg emits it.

    Returns:
        A 1-D ``np.float32`` array of samples in ``[-1, 1]``.

    Raises:
        FfmpegNotFoundError: If the ffmpeg executable can't be found.
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If ffmpeg fails to decode the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    command = [
        _ffmpeg_path(),
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sr),
    ]
    if mono:
        command += ["-ac", "1"]
    command += ["-"]

    try:
        proc = subprocess.run(
            command, capture_output=True, check=False, timeout=FFMPEG_TIMEOUT_SECONDS
        )
    except FileNotFoundError as exc:
        raise FfmpegNotFoundError(
            "the 'ffmpeg' executable was not found on PATH. Install it "
            "(Debian/Ubuntu: 'sudo apt-get install ffmpeg', macOS: "
            "'brew install ffmpeg', Windows: 'winget install Gyan.FFmpeg') "
            f"or set {FFMPEG_COMMAND_ENVVAR} to its path."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out after {FFMPEG_TIMEOUT_SECONDS:.0f}s decoding {path!r}"
        ) from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path!r}: {stderr}")

    return np.frombuffer(proc.stdout, dtype=np.float32)
