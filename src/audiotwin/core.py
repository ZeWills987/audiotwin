"""Core audio-twin detection pipeline.

Three levels, from cheapest to most involved:

* Level 0 — raw file hash (SHA256). Trivial exact-match short circuit.
* Level 1 — Chromaprint acoustic fingerprint + bit-wise similarity.
* Level 2 — NFP (neural fingerprint) score, computed *by the caller* and
  merged into the final verdict. This library never computes neural
  embeddings itself — that keeps the dependency footprint tiny.
"""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess

import numpy as np

# The level-1 fingerprint functions shell out to the ``fpcalc`` CLI (the
# Chromaprint command-line tool) in "-raw" mode, which prints the
# uncompressed fingerprint as plain integers. This avoids depending on the
# ``chromaprint`` ctypes binding, which requires a shared library
# (``libchromaprint``/``chromaprint.dll``) that upstream no longer ships for
# Windows — ``fpcalc`` alone (statically linked, bundling its own ffmpeg) is
# enough, and it's the same binary already documented as a system dependency.

#: Name of (or path to) the fpcalc executable. Override via the
#: ``AUDIOTWIN_FPCALC`` environment variable if it isn't on PATH.
FPCALC_COMMAND_ENVVAR = "AUDIOTWIN_FPCALC"
FPCALC_COMMAND = "fpcalc"

#: Minimum audio duration (seconds) required to compute a fingerprint.
MIN_DURATION_SECONDS = 10

#: Default Chromaprint similarity above which two tracks are the same master.
DEFAULT_CHROMAPRINT_THRESHOLD = 0.85

#: Default NFP similarity required to confirm a Chromaprint match.
DEFAULT_NFP_THRESHOLD = 0.90

# Number of bits per raw Chromaprint sub-fingerprint (uint32).
_BITS_PER_WORD = 32


class AudioTooShortError(ValueError):
    """Raised when an audio file is too short to fingerprint reliably."""


class FpcalcNotFoundError(RuntimeError):
    """Raised when the ``fpcalc`` (Chromaprint) executable can't be found."""


def file_hash(path: str) -> str:
    """Return the SHA256 hex digest of the raw file bytes.

    Two files with identical hashes are byte-for-byte identical and can be
    treated as duplicates without any acoustic analysis.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fpcalc_path() -> str:
    return os.environ.get(FPCALC_COMMAND_ENVVAR, FPCALC_COMMAND)


def _run_fpcalc(path: str, max_duration: int) -> tuple[float, np.ndarray]:
    """Run ``fpcalc -raw`` and parse its ``DURATION=``/``FINGERPRINT=`` output.

    ``-raw`` mode prints the uncompressed fingerprint as plain comma-separated
    32-bit integers, so no separate decompression step (and no
    ``libchromaprint`` shared library) is required.
    """
    command = [_fpcalc_path(), "-raw", "-length", str(max_duration), path]
    try:
        proc = subprocess.run(command, capture_output=True, check=False, text=True)
    except FileNotFoundError as exc:
        raise FpcalcNotFoundError(
            "the 'fpcalc' executable (Chromaprint) was not found on PATH. "
            f"Install it (see README) or set {FPCALC_COMMAND_ENVVAR} to its path."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(f"fpcalc failed on {path!r}: {proc.stderr.strip()}")

    duration = None
    fingerprint_ints = None
    for line in proc.stdout.splitlines():
        if line.startswith("DURATION="):
            duration = float(line[len("DURATION=") :])
        elif line.startswith("FINGERPRINT="):
            raw = line[len("FINGERPRINT=") :]
            fingerprint_ints = np.array(raw.split(","), dtype=np.int32)

    if duration is None or fingerprint_ints is None:
        raise RuntimeError(f"unexpected fpcalc output for {path!r}: {proc.stdout!r}")

    return duration, fingerprint_ints


def compute_fingerprint(path: str, max_duration: int = 120) -> str:
    """Compute a Chromaprint fingerprint for the first ``max_duration`` seconds.

    Decoding (to mono, 44100 Hz) and fingerprinting are both handled by the
    ``fpcalc`` command-line tool, which bundles its own ffmpeg-based decoder.

    Args:
        path: Path to the audio file.
        max_duration: Number of leading seconds to fingerprint. Default 120.

    Returns:
        A base64-encoded string of the raw fingerprint's 32-bit words.

    Raises:
        AudioTooShortError: If the decoded audio is shorter than
            :data:`MIN_DURATION_SECONDS` seconds.
        FpcalcNotFoundError: If the ``fpcalc`` executable can't be found.
    """
    duration, fingerprint_ints = _run_fpcalc(path, max_duration)

    if duration < MIN_DURATION_SECONDS:
        raise AudioTooShortError(
            f"{path!r} is {duration:.1f}s long; at least "
            f"{MIN_DURATION_SECONDS}s of audio is required to fingerprint."
        )

    return base64.b64encode(fingerprint_ints.tobytes()).decode("ascii")


def _decode(fingerprint: str) -> np.ndarray:
    """Decode an audiotwin fingerprint string back into an int32 word array."""
    raw = base64.b64decode(fingerprint)
    return np.frombuffer(raw, dtype=np.int32)


def compare_fingerprints(fp_a: str, fp_b: str) -> float:
    """Compare two Chromaprint fingerprints bit-by-bit.

    This is the standard AcoustID similarity: decode both fingerprints into
    their raw 32-bit sub-fingerprint arrays, XOR the overlapping prefix, and
    return ``1 - (differing_bits / total_bits)``.

    Args:
        fp_a: First fingerprint, as returned by :func:`compute_fingerprint`.
        fp_b: Second fingerprint, as returned by :func:`compute_fingerprint`.

    Returns:
        Similarity in ``[0.0, 1.0]`` — ``1.0`` means identical.
    """
    words_a = _decode(fp_a).view(np.uint32)
    words_b = _decode(fp_b).view(np.uint32)

    n = min(len(words_a), len(words_b))
    if n == 0:
        return 0.0

    xor = np.bitwise_xor(words_a[:n], words_b[:n])
    differing_bits = int(np.unpackbits(xor.view(np.uint8)).sum())
    total_bits = n * _BITS_PER_WORD
    return 1.0 - differing_bits / total_bits


def combine_scores(
    chromaprint_score: float,
    nfp_score: float | None = None,
    nfp_segments_matched: int | None = None,
    nfp_coverage: float | None = None,
    *,
    file_hash_match: bool = False,
    chromaprint_threshold: float = DEFAULT_CHROMAPRINT_THRESHOLD,
    nfp_threshold: float = DEFAULT_NFP_THRESHOLD,
) -> dict:
    """Merge the level-0/1/2 signals into a final duplicate verdict.

    Decision logic:

    * ``file_hash_match`` is ``True`` → duplicate, ``confidence = 1.0``
      (short-circuits everything else).
    * Otherwise, if ``chromaprint_score >= chromaprint_threshold``:

      - ``nfp_score`` provided and ``>= nfp_threshold`` → duplicate,
        ``confidence = (chromaprint_score + nfp_score) / 2``.
      - ``nfp_score`` **not** provided → duplicate,
        ``confidence = chromaprint_score * 0.9`` (slight penalty for the
        missing second confirmation).
      - ``nfp_score`` provided but ``< nfp_threshold`` → **not** a
        duplicate, ``confidence = 0.0``. This is the anti-false-positive
        guard: when the neural fingerprint contradicts Chromaprint, we
        reject the match.

    * Otherwise → not a duplicate, ``confidence = 0.0``.

    Args:
        chromaprint_score: Chromaprint similarity in ``[0, 1]``.
        nfp_score: Optional caller-provided neural-fingerprint similarity.
        nfp_segments_matched: Optional NFP metadata, passed through untouched.
        nfp_coverage: Optional NFP metadata, passed through untouched.
        file_hash_match: Whether the two files hash to the same value.
        chromaprint_threshold: Chromaprint match threshold (default 0.85).
        nfp_threshold: NFP confirmation threshold (default 0.90).

    Returns:
        A dict with ``chromaprint_score``, ``nfp_score``,
        ``nfp_segments_matched``, ``nfp_coverage``, ``file_hash_match``,
        ``is_duplicate`` and ``confidence``.
    """
    result = {
        "file_hash_match": file_hash_match,
        "chromaprint_score": chromaprint_score,
        "nfp_score": nfp_score,
        "nfp_segments_matched": nfp_segments_matched,
        "nfp_coverage": nfp_coverage,
        "is_duplicate": False,
        "confidence": 0.0,
    }

    if file_hash_match:
        result["is_duplicate"] = True
        result["confidence"] = 1.0
        return result

    if chromaprint_score >= chromaprint_threshold:
        if nfp_score is None:
            result["is_duplicate"] = True
            result["confidence"] = chromaprint_score * 0.9
        elif nfp_score >= nfp_threshold:
            result["is_duplicate"] = True
            result["confidence"] = (chromaprint_score + nfp_score) / 2
        # else: NFP contradicts Chromaprint → reject (defaults stand).

    return result


def detect(
    path_a: str,
    path_b: str,
    nfp_score: float | None = None,
    nfp_segments_matched: int | None = None,
    nfp_coverage: float | None = None,
    *,
    max_duration: int = 120,
    chromaprint_threshold: float = DEFAULT_CHROMAPRINT_THRESHOLD,
    nfp_threshold: float = DEFAULT_NFP_THRESHOLD,
) -> dict:
    """Detect whether two audio files share the same master recording.

    Runs the pipeline cheapest-first: a raw file hash short-circuits the whole
    thing, otherwise Chromaprint fingerprints are computed and compared, and
    finally an optional caller-supplied NFP score is merged in.

    Args:
        path_a: Path to the first audio file.
        path_b: Path to the second audio file.
        nfp_score: Optional precomputed neural-fingerprint similarity.
        nfp_segments_matched: Optional NFP metadata, passed through.
        nfp_coverage: Optional NFP metadata, passed through.
        max_duration: Seconds of leading audio to fingerprint (default 120).
        chromaprint_threshold: Chromaprint match threshold (default 0.85).
        nfp_threshold: NFP confirmation threshold (default 0.90).

    Returns:
        A verdict dict (see module docstring / README for the schema).
    """
    hash_match = file_hash(path_a) == file_hash(path_b)

    if hash_match:
        result = combine_scores(
            chromaprint_score=1.0,
            nfp_score=nfp_score,
            nfp_segments_matched=nfp_segments_matched,
            nfp_coverage=nfp_coverage,
            file_hash_match=True,
            chromaprint_threshold=chromaprint_threshold,
            nfp_threshold=nfp_threshold,
        )
    else:
        fp_a = compute_fingerprint(path_a, max_duration=max_duration)
        fp_b = compute_fingerprint(path_b, max_duration=max_duration)
        chromaprint_score = compare_fingerprints(fp_a, fp_b)
        result = combine_scores(
            chromaprint_score=chromaprint_score,
            nfp_score=nfp_score,
            nfp_segments_matched=nfp_segments_matched,
            nfp_coverage=nfp_coverage,
            file_hash_match=False,
            chromaprint_threshold=chromaprint_threshold,
            nfp_threshold=nfp_threshold,
        )

    return {"track_a": path_a, "track_b": path_b, **result}
