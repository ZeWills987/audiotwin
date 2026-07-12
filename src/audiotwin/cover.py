"""Cover-song similarity via chroma + optimal transposition + DTW.

A simplified take on the classic Serrà 2009 approach: beat-agnostic chroma
features, Optimal Transposition Index (OTI) to align keys, and dynamic time
warping to absorb tempo differences. Supports COVER/LIVE/ACOUSTIC-style
relations; the fine distinction between those stays with the caller.

Requires the ``[cover]`` extra (librosa). All heavy imports are lazy so the
core install stays light.
"""

from __future__ import annotations

import numpy as np

from audiotwin.audio import decode_audio

#: Default decode/analysis sample rate for chroma features.
DEFAULT_COVER_SR = 22050

#: Default chroma frames per second after temporal aggregation.
DEFAULT_TARGET_FPS = 2.0

#: Default Sakoe-Chiba band as a fraction of the longer sequence.
DEFAULT_DTW_BAND_RATIO = 0.25


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "the cover module requires librosa — install it with "
            "'pip install audiotwin[cover]'"
        ) from exc
    return librosa


def compute_chroma(
    audio: np.ndarray,
    sr: int = DEFAULT_COVER_SR,
    target_fps: float = DEFAULT_TARGET_FPS,
    use_hpss: bool = True,
) -> np.ndarray:
    """Compute an L2-normalized, time-aggregated chroma matrix.

    Pipeline: optional harmonic/percussive separation (keeping the harmonic
    component) → CQT chroma → temporal mean-aggregation down to
    ``target_fps`` frames per second → per-column L2 normalization.

    Args:
        audio: Mono float32 PCM (output of :func:`audiotwin.audio.decode_audio`).
        sr: Sample rate of ``audio`` (default 22050).
        target_fps: Chroma frames per second after aggregation (default 2.0).
        use_hpss: Isolate the harmonic component first (default True);
            set to False to skip the (slow) separation.

    Returns:
        A ``(12, T)`` float array, each column L2-normalized.
    """
    librosa = _require_librosa()

    if use_hpss:
        audio = librosa.effects.harmonic(audio)

    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)

    native_fps = sr / hop_length
    block = max(1, int(round(native_fps / target_fps)))
    n_blocks = chroma.shape[1] // block
    if n_blocks == 0:
        aggregated = chroma.mean(axis=1, keepdims=True)
    else:
        aggregated = (
            chroma[:, : n_blocks * block].reshape(12, n_blocks, block).mean(axis=2)
        )

    norms = np.linalg.norm(aggregated, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return aggregated / norms


def optimal_transposition(chroma_a: np.ndarray, chroma_b: np.ndarray) -> tuple[int, float]:
    """Optimal Transposition Index between two chroma matrices.

    Tries the 12 circular rotations of ``chroma_b``'s mean profile against
    ``chroma_a``'s and returns the best one.

    The returned rotation ``k`` means: **track B sounds transposed UP by
    ``k`` semitones relative to track A** — rolling ``chroma_b`` by ``-k``
    along the pitch axis aligns it with ``chroma_a``.

    Args:
        chroma_a: ``(12, T_a)`` chroma matrix.
        chroma_b: ``(12, T_b)`` chroma matrix.

    Returns:
        ``(k, similarity)`` where ``k`` is in ``0..11`` and ``similarity``
        is the correlation of the mean chroma profiles at that rotation.
    """
    mean_a = chroma_a.mean(axis=1)
    mean_b = chroma_b.mean(axis=1)

    mean_a = mean_a - mean_a.mean()
    mean_b = mean_b - mean_b.mean()
    norm_a = np.linalg.norm(mean_a) or 1.0
    norm_b = np.linalg.norm(mean_b) or 1.0

    best_k, best_sim = 0, -np.inf
    for k in range(12):
        sim = float(np.dot(mean_a, np.roll(mean_b, -k)) / (norm_a * norm_b))
        if sim > best_sim:
            best_k, best_sim = k, sim
    return best_k, best_sim


def _dtw_similarity(
    chroma_a: np.ndarray,
    chroma_b: np.ndarray,
    dtw_band_ratio: float,
) -> tuple[float, float, int, int]:
    """OTI-align then DTW two chroma matrices.

    Returns ``(similarity, normalized_cost, path_length, transposition)``.
    """
    librosa = _require_librosa()

    transposition, _ = optimal_transposition(chroma_a, chroma_b)
    chroma_b_aligned = np.roll(chroma_b, -transposition, axis=0)

    band_rad = max(dtw_band_ratio, 1e-3)
    try:
        cost_matrix, path = librosa.sequence.dtw(
            X=chroma_a,
            Y=chroma_b_aligned,
            metric="cosine",
            global_constraints=True,
            band_rad=band_rad,
        )
    except librosa.ParameterError:
        # The Sakoe-Chiba band can be infeasible when the two sequences have
        # very different lengths; retry unconstrained.
        cost_matrix, path = librosa.sequence.dtw(
            X=chroma_a, Y=chroma_b_aligned, metric="cosine"
        )

    path_length = len(path)
    normalized_cost = float(cost_matrix[-1, -1]) / max(path_length, 1)
    similarity = float(np.clip(1.0 - normalized_cost, 0.0, 1.0))
    return similarity, normalized_cost, path_length, transposition


def cover_similarity_from_chroma(
    chroma_a: np.ndarray,
    chroma_b: np.ndarray,
    dtw_band_ratio: float = DEFAULT_DTW_BAND_RATIO,
) -> dict:
    """Cover similarity from precomputed chroma matrices.

    For callers that cache their chroma features and don't want audiotwin
    to re-decode audio. Same output as :func:`cover_similarity` minus the
    path/duration fields.

    Args:
        chroma_a: ``(12, T_a)`` chroma matrix (see :func:`compute_chroma`).
        chroma_b: ``(12, T_b)`` chroma matrix.
        dtw_band_ratio: Sakoe-Chiba band radius as a fraction of the longer
            sequence (default 0.25).

    Returns:
        A dict with ``similarity``, ``transposition_semitones``,
        ``dtw_normalized_cost`` and ``path_length``.
    """
    similarity, normalized_cost, path_length, transposition = _dtw_similarity(
        chroma_a, chroma_b, dtw_band_ratio
    )
    return {
        "similarity": similarity,
        "transposition_semitones": transposition,
        "dtw_normalized_cost": normalized_cost,
        "path_length": path_length,
    }


def cover_similarity(
    path_a: str,
    path_b: str,
    sr: int = DEFAULT_COVER_SR,
    target_fps: float = DEFAULT_TARGET_FPS,
    use_hpss: bool = True,
    dtw_band_ratio: float = DEFAULT_DTW_BAND_RATIO,
) -> dict:
    """Composition-level similarity between two audio files.

    Pipeline: in-memory decode → chroma ×2 → OTI transposition alignment →
    DTW with a Sakoe-Chiba band → path-length-normalized cost.

    Args:
        path_a: Path to the first audio file.
        path_b: Path to the second audio file.
        sr: Analysis sample rate (default 22050).
        target_fps: Chroma frames per second (default 2.0).
        use_hpss: Isolate harmonics before chroma (default True).
        dtw_band_ratio: Sakoe-Chiba band radius, relative (default 0.25).

    Returns:
        A dict with ``track_a``, ``track_b``, ``similarity`` (``0.0–1.0``),
        ``transposition_semitones`` (``0–11``, how far B is transposed up
        relative to A), ``dtw_normalized_cost``, ``path_length`` and
        ``duration_ratio`` (``duration_b / duration_a`` — a free global
        tempo hint).
    """
    audio_a = decode_audio(path_a, sr=sr)
    audio_b = decode_audio(path_b, sr=sr)

    chroma_a = compute_chroma(audio_a, sr=sr, target_fps=target_fps, use_hpss=use_hpss)
    chroma_b = compute_chroma(audio_b, sr=sr, target_fps=target_fps, use_hpss=use_hpss)

    result = cover_similarity_from_chroma(chroma_a, chroma_b, dtw_band_ratio=dtw_band_ratio)
    duration_a = len(audio_a) / sr
    duration_b = len(audio_b) / sr

    return {
        "track_a": path_a,
        "track_b": path_b,
        **result,
        "duration_ratio": duration_b / duration_a if duration_a else 0.0,
    }
