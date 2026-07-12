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

#: Default Chromaprint similarity range (inclusive lower bound) considered
#: for a possible REMASTER, below the duplicate threshold.
DEFAULT_REMASTER_CHROMAPRINT_MIN = 0.60

#: Default NFP similarity required to confirm a REMASTER.
DEFAULT_REMASTER_NFP_THRESHOLD = 0.90

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


def classify_relation(
    chromaprint_score: float,
    nfp_score: float | None = None,
    duplicate_threshold: float = DEFAULT_CHROMAPRINT_THRESHOLD,
    remaster_chromaprint_min: float = DEFAULT_REMASTER_CHROMAPRINT_MIN,
    remaster_chromaprint_max: float = DEFAULT_CHROMAPRINT_THRESHOLD,
    remaster_nfp_threshold: float = DEFAULT_REMASTER_NFP_THRESHOLD,
) -> dict:
    """Classify the relation between two tracks from their similarity scores.

    REMASTER is the signature of NFP staying high while Chromaprint drops —
    same structural content, different spectral texture (EQ, dynamics, a
    light re-mix). It reuses the exact same two scores as :func:`detect` /
    :func:`combine_scores`; no new audio analysis is involved.

    Decision grid:

    * ``chromaprint_score >= duplicate_threshold`` → ``"DUPLICATE"``
      (regardless of ``nfp_score``). Confidence follows the same logic as
      :func:`combine_scores`.
    * ``remaster_chromaprint_min <= chromaprint_score < duplicate_threshold``
      and ``nfp_score >= remaster_nfp_threshold`` → ``"REMASTER"``,
      ``confidence = nfp_score * 0.9`` (a slight penalty: REMASTER is
      inherently a less certain call than DUPLICATE — the chromaprint/nfp
      gap could also come from a Chromaprint false positive on a very
      repetitive track).
    * ``remaster_chromaprint_min <= chromaprint_score < duplicate_threshold``
      and (``nfp_score < remaster_nfp_threshold`` or ``nfp_score is None``)
      → ``"NO_RELATION"`` (an unconfirmed fingerprint coincidence),
      ``confidence = 0.0``.
    * ``chromaprint_score < remaster_chromaprint_min`` → ``"NO_RELATION"``,
      ``confidence = 0.0``, even when ``nfp_score`` is high: below this
      floor the spectral link is considered too thin to trust, whatever NFP
      suggests about structural similarity.

    Args:
        chromaprint_score: Chromaprint similarity in ``[0, 1]``.
        nfp_score: Optional caller-provided neural-fingerprint similarity.
        duplicate_threshold: Chromaprint match threshold (default 0.85).
        remaster_chromaprint_min: Lower Chromaprint bound considered for a
            REMASTER (default 0.60).
        remaster_chromaprint_max: Upper Chromaprint bound considered for a
            REMASTER, i.e. the duplicate threshold (default 0.85).
        remaster_nfp_threshold: NFP confirmation threshold for REMASTER
            (default 0.90).

    Returns:
        A dict with ``relation_type`` (``"DUPLICATE"`` | ``"REMASTER"`` |
        ``"NO_RELATION"``), ``chromaprint_score``, ``nfp_score``,
        ``score_gap`` (``nfp_score - chromaprint_score``, or ``None`` if
        ``nfp_score`` is ``None``) and ``confidence``.
    """
    score_gap = None if nfp_score is None else nfp_score - chromaprint_score

    if chromaprint_score >= duplicate_threshold:
        duplicate = combine_scores(chromaprint_score, nfp_score)
        return {
            "relation_type": "DUPLICATE",
            "chromaprint_score": chromaprint_score,
            "nfp_score": nfp_score,
            "score_gap": score_gap,
            "confidence": duplicate["confidence"],
        }

    if remaster_chromaprint_min <= chromaprint_score < remaster_chromaprint_max:
        if nfp_score is not None and nfp_score >= remaster_nfp_threshold:
            return {
                "relation_type": "REMASTER",
                "chromaprint_score": chromaprint_score,
                "nfp_score": nfp_score,
                "score_gap": score_gap,
                "confidence": nfp_score * 0.9,
            }

    return {
        "relation_type": "NO_RELATION",
        "chromaprint_score": chromaprint_score,
        "nfp_score": nfp_score,
        "score_gap": score_gap,
        "confidence": 0.0,
    }


def detect_relation(
    path_a: str,
    path_b: str,
    skip_decode_if_hash_match: bool = True,
    nfp_score: float | None = None,
    nfp_segments_matched: int | None = None,
    nfp_coverage: float | None = None,
    *,
    max_duration: int = 120,
    duplicate_threshold: float = DEFAULT_CHROMAPRINT_THRESHOLD,
    remaster_chromaprint_min: float = DEFAULT_REMASTER_CHROMAPRINT_MIN,
    remaster_chromaprint_max: float = DEFAULT_CHROMAPRINT_THRESHOLD,
    remaster_nfp_threshold: float = DEFAULT_REMASTER_NFP_THRESHOLD,
) -> dict:
    """Classify whether two audio files are a DUPLICATE, REMASTER, or unrelated.

    Same pipeline shape as :func:`detect`, but calls :func:`classify_relation`
    instead of :func:`combine_scores`. A file hash match short-circuits
    straight to ``relation_type="DUPLICATE"``, ``confidence=1.0``, without
    ever computing a Chromaprint fingerprint (unless
    ``skip_decode_if_hash_match`` is set to ``False``).

    Args:
        path_a: Path to the first audio file.
        path_b: Path to the second audio file.
        skip_decode_if_hash_match: When ``True`` (default), a file hash match
            short-circuits straight to DUPLICATE without computing
            Chromaprint fingerprints. Set to ``False`` to always run the
            fingerprint/classification pipeline regardless of hash match.
        nfp_score: Optional precomputed neural-fingerprint similarity.
        nfp_segments_matched: Optional NFP metadata, passed through.
        nfp_coverage: Optional NFP metadata, passed through.
        max_duration: Seconds of leading audio to fingerprint (default 120).
        duplicate_threshold: Chromaprint match threshold (default 0.85).
        remaster_chromaprint_min: Lower Chromaprint bound for REMASTER
            (default 0.60).
        remaster_chromaprint_max: Upper Chromaprint bound for REMASTER
            (default 0.85).
        remaster_nfp_threshold: NFP confirmation threshold for REMASTER
            (default 0.90).

    Returns:
        A dict with ``track_a``, ``track_b``, ``file_hash_match``,
        ``chromaprint_score``, ``nfp_score``, ``nfp_segments_matched``,
        ``nfp_coverage``, ``score_gap``, ``relation_type`` and ``confidence``.
    """
    hash_match = file_hash(path_a) == file_hash(path_b)

    if hash_match and skip_decode_if_hash_match:
        relation = {
            "relation_type": "DUPLICATE",
            "chromaprint_score": 1.0,
            "nfp_score": nfp_score,
            "score_gap": None if nfp_score is None else nfp_score - 1.0,
            "confidence": 1.0,
        }
    else:
        fp_a = compute_fingerprint(path_a, max_duration=max_duration)
        fp_b = compute_fingerprint(path_b, max_duration=max_duration)
        chromaprint_score = compare_fingerprints(fp_a, fp_b)
        relation = classify_relation(
            chromaprint_score,
            nfp_score,
            duplicate_threshold=duplicate_threshold,
            remaster_chromaprint_min=remaster_chromaprint_min,
            remaster_chromaprint_max=remaster_chromaprint_max,
            remaster_nfp_threshold=remaster_nfp_threshold,
        )

    return {
        "track_a": path_a,
        "track_b": path_b,
        "file_hash_match": hash_match,
        "nfp_segments_matched": nfp_segments_matched,
        "nfp_coverage": nfp_coverage,
        **relation,
    }


# --- EDIT classification -----------------------------------------------------
#
# Unlike detect()/detect_relation(), the functions below never decode audio.
# They operate on match points (t_query, t_ref, score) already produced by an
# external segment-matching system (e.g. a neural fingerprinter or landmark
# matcher), and infer the temporal relation geometrically.
#
# RANSAC is implemented by hand with numpy rather than via
# sklearn.linear_model.RANSACRegressor: scikit-learn pulls in scipy, adding
# ~46 MB of wheels beyond numpy (measured: scipy 37 MB + sklearn 8 MB +
# joblib/threadpoolctl), which conflicts with this repo's install-in-seconds
# goal — for fitting a 2-parameter line, a manual implementation is tiny and
# dependency-free.

#: Default minimum number of inlier match points for a trustworthy fit.
DEFAULT_MIN_INLIERS = 6

#: Default plausible slope (speed factor) range for an edit relation.
DEFAULT_SLOPE_BOUNDS = (0.5, 2.0)

#: Default residual tolerance (seconds) for a point to count as an inlier.
DEFAULT_RESIDUAL_THRESHOLD = 0.5

#: Default slope deviation from 1.0 beyond which speed is considered changed.
DEFAULT_SPEED_CHANGE_EPSILON = 0.03

#: Default coverage above which a track side is considered fully covered.
DEFAULT_FULL_COVERAGE_THRESHOLD = 0.90

#: Default max gap (seconds) between consecutive inliers on the query axis
#: for the inlier set to still count as consecutive.
DEFAULT_MAX_CONSECUTIVE_GAP = 5.0


def fit_temporal_alignment(
    matches: list[tuple[float, float, float]],
    min_inliers: int = DEFAULT_MIN_INLIERS,
    slope_bounds: tuple[float, float] = DEFAULT_SLOPE_BOUNDS,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    ransac_iterations: int = 1000,
    random_seed: int | None = None,
) -> dict:
    """Fit ``t_ref = slope * t_query + intercept`` on match points via RANSAC.

    Each match is a ``(t_query, t_ref, match_score)`` triple from an external
    segment-matching system. The slope is the speed factor between the two
    tracks, the intercept the time offset.

    Args:
        matches: Match points as ``(t_query, t_ref, match_score)`` triples.
        min_inliers: Minimum inliers for the fit to succeed (default 6).
            When ``len(matches) < min_inliers``, the function fails fast
            without running RANSAC at all.
        slope_bounds: Plausible ``(min, max)`` slope range (default 0.5–2.0);
            fits outside it are rejected.
        residual_threshold: Max ``|t_ref - predicted|`` in seconds for a
            point to count as an inlier (default 0.5).
        ransac_iterations: Number of RANSAC sampling rounds (default 1000).
        random_seed: Optional seed for reproducible sampling.

    Returns:
        A dict with ``slope``, ``intercept``, ``inlier_count``,
        ``outlier_count``, ``inlier_indices`` and ``fit_succeeded``. On
        failure, ``slope``/``intercept`` are ``0.0`` and ``inlier_indices``
        is empty.
    """
    failure = {
        "slope": 0.0,
        "intercept": 0.0,
        "inlier_count": 0,
        "outlier_count": len(matches),
        "inlier_indices": [],
        "fit_succeeded": False,
    }

    if len(matches) < min_inliers:
        return failure

    points = np.asarray(matches, dtype=np.float64)
    t_query, t_ref = points[:, 0], points[:, 1]
    n = len(points)
    rng = np.random.default_rng(random_seed)
    slope_min, slope_max = slope_bounds

    best_inliers: np.ndarray | None = None
    best_count = 0

    for _ in range(ransac_iterations):
        i, j = rng.choice(n, size=2, replace=False)
        dt = t_query[j] - t_query[i]
        if dt == 0.0:
            continue
        slope = (t_ref[j] - t_ref[i]) / dt
        if not (slope_min <= slope <= slope_max):
            continue
        intercept = t_ref[i] - slope * t_query[i]
        residuals = np.abs(t_ref - (slope * t_query + intercept))
        inliers = residuals <= residual_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        return failure

    # Refine with a least-squares fit on the consensus set, then recompute
    # the inlier set against the refined line.
    slope, intercept = np.polyfit(t_query[best_inliers], t_ref[best_inliers], deg=1)
    if not (slope_min <= slope <= slope_max):
        return failure

    residuals = np.abs(t_ref - (slope * t_query + intercept))
    inliers = residuals <= residual_threshold
    inlier_count = int(inliers.sum())
    if inlier_count < min_inliers:
        return failure

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "inlier_count": inlier_count,
        "outlier_count": n - inlier_count,
        "inlier_indices": [int(k) for k in np.flatnonzero(inliers)],
        "fit_succeeded": True,
    }


def compute_coverage(
    matches: list[tuple[float, float, float]],
    inlier_indices: list[int],
    query_duration: float,
    ref_duration: float,
    max_consecutive_gap: float = DEFAULT_MAX_CONSECUTIVE_GAP,
) -> dict:
    """Compute how much of each track the inlier match points span.

    Coverage is the min→max temporal extent of the inlier points on each
    axis, divided by that track's total duration.

    Args:
        matches: Match points as ``(t_query, t_ref, match_score)`` triples.
        inlier_indices: Indices into ``matches`` of the inlier points (as
            returned by :func:`fit_temporal_alignment`).
        query_duration: Total duration of the query track, in seconds.
        ref_duration: Total duration of the reference track, in seconds.
        max_consecutive_gap: Max gap in seconds between consecutive inliers
            (sorted by ``t_query``) for ``is_consecutive`` to hold
            (default 5.0).

    Returns:
        A dict with ``coverage_query``, ``coverage_ref`` (both ``0.0–1.0``)
        and ``is_consecutive``.
    """
    if not inlier_indices:
        return {"coverage_query": 0.0, "coverage_ref": 0.0, "is_consecutive": False}

    points = np.asarray(matches, dtype=np.float64)[inlier_indices]
    t_query, t_ref = points[:, 0], points[:, 1]

    coverage_query = float(t_query.max() - t_query.min()) / query_duration
    coverage_ref = float(t_ref.max() - t_ref.min()) / ref_duration

    sorted_query = np.sort(t_query)
    gaps = np.diff(sorted_query)
    is_consecutive = bool(len(gaps) == 0 or gaps.max() <= max_consecutive_gap)

    return {
        "coverage_query": min(1.0, coverage_query),
        "coverage_ref": min(1.0, coverage_ref),
        "is_consecutive": is_consecutive,
    }


def classify_edit(
    matches: list[tuple[float, float, float]],
    query_duration: float,
    ref_duration: float,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    slope_bounds: tuple[float, float] = DEFAULT_SLOPE_BOUNDS,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    speed_change_epsilon: float = DEFAULT_SPEED_CHANGE_EPSILON,
    full_coverage_threshold: float = DEFAULT_FULL_COVERAGE_THRESHOLD,
    *,
    ransac_iterations: int = 1000,
    random_seed: int | None = None,
) -> dict:
    """Derive an edit-type hint from segment match points.

    Combines :func:`fit_temporal_alignment` and :func:`compute_coverage`.
    No audio is decoded — the inputs are match points already produced by an
    external segment-matching system.

    Decision grid:

    * fit failed → ``"no_relation"``
    * ``|slope - 1| > speed_change_epsilon`` → ``"speed_change"``
      (sped-up/slowed: tempo and pitch move together)
    * slope ≈ 1 and either side's coverage below
      ``full_coverage_threshold`` → ``"trim_or_extend"``
      (radio edit, extended version, ...)
    * slope ≈ 1 and both coverages ≥ ``full_coverage_threshold``
      → ``"full_match"``. This means **no edit detected — the temporal
      structure is intact**; the pair is better characterized as
      DUPLICATE/REMASTER territory, which the caller should corroborate
      via :func:`classify_relation`.

    Confidence: ``0.0`` for ``"no_relation"``; otherwise
    ``min(1.0, inlier_ratio * (inlier_count / min_inliers))``, i.e. the
    inlier proportion scaled by how far past the bare minimum the absolute
    inlier count goes. This deliberately favors "20 inliers out of 25"
    (ratio 0.8, evidence factor 20/6 → capped 1.0) over "6 inliers out of
    6" (perfect ratio but evidence factor exactly 1.0 → confidence 1.0
    only because both factors are maxed; with any outliers at few points,
    confidence drops fast) — more absolute evidence beats a perfect ratio
    on a tiny sample.

    Args:
        matches: Match points as ``(t_query, t_ref, match_score)`` triples.
        query_duration: Total duration of the query track, in seconds.
        ref_duration: Total duration of the reference track, in seconds.
        min_inliers: Minimum inliers for a trustworthy fit (default 6).
        slope_bounds: Plausible slope range (default 0.5–2.0).
        residual_threshold: Inlier tolerance in seconds (default 0.5).
        speed_change_epsilon: Slope deviation from 1.0 beyond which speed is
            considered changed (default 0.03, i.e. 3%).
        full_coverage_threshold: Coverage above which a side counts as fully
            covered (default 0.90).
        ransac_iterations: RANSAC sampling rounds (default 1000).
        random_seed: Optional seed for reproducible sampling.

    Returns:
        A dict with ``slope``, ``intercept``, ``inlier_count``,
        ``outlier_count``, ``coverage_query``, ``coverage_ref``,
        ``is_consecutive``, ``edit_type_hint`` (``"speed_change"`` |
        ``"trim_or_extend"`` | ``"full_match"`` | ``"no_relation"``) and
        ``confidence``.
    """
    fit = fit_temporal_alignment(
        matches,
        min_inliers=min_inliers,
        slope_bounds=slope_bounds,
        residual_threshold=residual_threshold,
        ransac_iterations=ransac_iterations,
        random_seed=random_seed,
    )
    coverage = compute_coverage(
        matches, fit["inlier_indices"], query_duration, ref_duration
    )

    if not fit["fit_succeeded"]:
        edit_type_hint = "no_relation"
        confidence = 0.0
    else:
        if abs(fit["slope"] - 1.0) > speed_change_epsilon:
            edit_type_hint = "speed_change"
        elif (
            coverage["coverage_query"] < full_coverage_threshold
            or coverage["coverage_ref"] < full_coverage_threshold
        ):
            edit_type_hint = "trim_or_extend"
        else:
            edit_type_hint = "full_match"

        total = fit["inlier_count"] + fit["outlier_count"]
        inlier_ratio = fit["inlier_count"] / total
        confidence = min(1.0, inlier_ratio * (fit["inlier_count"] / min_inliers))

    return {
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "inlier_count": fit["inlier_count"],
        "outlier_count": fit["outlier_count"],
        "coverage_query": coverage["coverage_query"],
        "coverage_ref": coverage["coverage_ref"],
        "is_consecutive": coverage["is_consecutive"],
        "edit_type_hint": edit_type_hint,
        "confidence": confidence,
    }


# --- INSTRUMENTAL / KARAOKE pair classification ------------------------------

#: Default content-similarity floor for an instrumental pair.
DEFAULT_CONTENT_THRESHOLD = 0.70

#: Default vocal coverage above which a track counts as "has vocals".
DEFAULT_VOCAL_PRESENT_THRESHOLD = 0.40

#: Default vocal coverage below which a track counts as "no vocals".
DEFAULT_VOCAL_ABSENT_THRESHOLD = 0.10


def classify_instrumental_pair(
    content_similarity: float,
    vocal_coverage_a: float,
    vocal_coverage_b: float,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    vocal_present_threshold: float = DEFAULT_VOCAL_PRESENT_THRESHOLD,
    vocal_absent_threshold: float = DEFAULT_VOCAL_ABSENT_THRESHOLD,
) -> dict:
    """Detect the instrumental/karaoke pattern between two tracks.

    The pattern: same musical content, one track has vocals, the other has
    (almost) none. Pure score arithmetic — no audio is decoded and no new
    dependency is involved.

    ``content_similarity`` can be ANY ``[0, 1]`` measure of "same musical
    content": the ``similarity`` from :func:`audiotwin.cover.cover_similarity`,
    a landmark score, an NFP score — whatever the caller trusts. The vocal
    coverages are measured by an external vocal-activity detector (this
    library deliberately doesn't ship one).

    Args:
        content_similarity: ``[0, 1]`` same-musical-content measure.
        vocal_coverage_a: ``[0, 1]`` share of track A with detected vocals.
        vocal_coverage_b: ``[0, 1]`` share of track B with detected vocals.
        content_threshold: Content-similarity floor (default 0.70).
        vocal_present_threshold: Coverage above which a track "has vocals"
            (default 0.40).
        vocal_absent_threshold: Coverage below which a track "has no vocals"
            (default 0.10).

    Returns:
        A dict with ``is_instrumental_pair``, ``vocal_track`` /
        ``instrumental_track`` (``"a"`` | ``"b"`` | ``None``),
        ``content_similarity``, ``vocal_gap``
        (``|vocal_coverage_a - vocal_coverage_b|``) and ``confidence``
        (``content_similarity * vocal_gap`` when the pattern is detected,
        else ``0.0``).
    """
    vocal_gap = abs(vocal_coverage_a - vocal_coverage_b)

    a_vocal_b_instrumental = (
        vocal_coverage_a >= vocal_present_threshold
        and vocal_coverage_b <= vocal_absent_threshold
    )
    b_vocal_a_instrumental = (
        vocal_coverage_b >= vocal_present_threshold
        and vocal_coverage_a <= vocal_absent_threshold
    )

    is_pair = content_similarity >= content_threshold and (
        a_vocal_b_instrumental or b_vocal_a_instrumental
    )

    if is_pair:
        vocal_track = "a" if a_vocal_b_instrumental else "b"
        instrumental_track = "b" if a_vocal_b_instrumental else "a"
        confidence = content_similarity * vocal_gap
    else:
        vocal_track = None
        instrumental_track = None
        confidence = 0.0

    return {
        "is_instrumental_pair": is_pair,
        "vocal_track": vocal_track,
        "instrumental_track": instrumental_track,
        "content_similarity": content_similarity,
        "vocal_gap": vocal_gap,
        "confidence": confidence,
    }
