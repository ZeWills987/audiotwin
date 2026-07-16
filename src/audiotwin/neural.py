"""Neural fingerprinting bridge — Sony Sample-ID embeddings.

Wraps the MIT-licensed Sample-ID model (Riou, Serrà & Mitsufuji, ICASSP
2026; https://github.com/sony/sampleid) as the neural signal the rest of
audiotwin was designed to accept but never computes itself:

* :func:`neural_similarity` returns exactly the ``nfp_score`` /
  ``nfp_segments_matched`` / ``nfp_coverage`` triple that
  :func:`audiotwin.core.detect`, :func:`audiotwin.core.detect_relation`
  and :func:`audiotwin.core.combine_scores` take as optional inputs —
  unlocking REMASTER classification without any external system.
* :func:`neural_match_points` returns ``(t_query, t_ref, score)`` triples
  in the exact format :func:`audiotwin.core.classify_edit` and
  :func:`audiotwin.core.fit_temporal_alignment` consume — a matcher that
  survives the transformations (EQ, overdubs, pitch/time edits) that
  break the classical landmark front-end.

Requires the ``[neural]`` extra (PyTorch) plus the ``sampleid`` package
installed from GitHub (it is not on PyPI)::

    pip install "audiotwin[neural]"
    pip install -e "git+https://github.com/sony/sampleid.git#egg=sampleid"

The pretrained checkpoint (MIT, Zenodo record 17413869) is downloaded by
audiotwin on first use into ``~/.cache/audiotwin`` (override with the
``AUDIOTWIN_CACHE`` environment variable) — audiotwin redistributes
neither the model code nor the weights.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

from audiotwin.audio import decode_audio

#: Sample rate the Sample-ID model expects.
NEURAL_SR = 16000

#: Default chunk length in seconds (the model averages embeddings over its
#: input, so full-track similarity needs manual chunking).
DEFAULT_CHUNK_SECONDS = 5.0

#: Default hop between chunks in seconds (50% overlap).
DEFAULT_CHUNK_HOP_SECONDS = 2.5

#: Default cosine similarity above which a query chunk counts as matched.
DEFAULT_MATCH_THRESHOLD = 0.60

#: Default batch size for embedding extraction.
DEFAULT_BATCH_SIZE = 16

#: Override the checkpoint cache directory via this environment variable.
CACHE_DIR_ENVVAR = "AUDIOTWIN_CACHE"

_MODEL_CACHE: dict = {}


def _cache_dir() -> str:
    return os.environ.get(
        CACHE_DIR_ENVVAR, os.path.join(os.path.expanduser("~"), ".cache", "audiotwin")
    )


def _default_checkpoint(SampleID) -> str:
    """Download (once) the official Zenodo checkpoint into audiotwin's cache.

    audiotwin handles the download itself instead of delegating to
    ``sampleid.load_checkpoint()``: upstream's downloader moves a still-open
    ``NamedTemporaryFile(delete=True)`` into place, and on Windows the
    delete-on-close flag follows the handle — the moved checkpoint vanishes
    when the tempfile closes. Downloading to ``<target>.part`` then
    ``os.replace`` is atomic and cross-platform, and keeps the 805 MB file
    in a user cache instead of site-packages.
    """
    target = os.path.join(_cache_dir(), "sampleid-best.ckpt")
    if os.path.exists(target):
        return target

    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".part"
    urllib.request.urlretrieve(SampleID.default_ckpt_url, partial)
    os.replace(partial, target)
    return target


def _require_sampleid():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the neural module requires PyTorch — install it with "
            "'pip install audiotwin[neural]'"
        ) from exc
    try:
        from sampleid import SampleID
    except ImportError as exc:
        raise ImportError(
            "the neural module requires the 'sampleid' package (MIT, not on "
            "PyPI) — install it in EDITABLE mode (upstream packaging omits "
            "its src/ subpackage in regular installs): pip install -e "
            '"git+https://github.com/sony/sampleid.git#egg=sampleid"'
        ) from exc
    return SampleID


def _load_model(checkpoint_path: str | None = None):
    """Load (and memoize) the Sample-ID model in eval mode."""
    import torch

    SampleID = _require_sampleid()
    key = checkpoint_path or "__default__"
    if key not in _MODEL_CACHE:
        resolved = checkpoint_path or _default_checkpoint(SampleID)
        model = SampleID.load_checkpoint(ckpt_path=resolved)
        model.eval()
        torch.set_grad_enabled(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def neural_embedding(
    path: str,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_hop_seconds: float = DEFAULT_CHUNK_HOP_SECONDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_path: str | None = None,
) -> np.ndarray:
    """Per-chunk Sample-ID embeddings for an audio file.

    The track is decoded to 16 kHz mono in memory, split into overlapping
    chunks, and each chunk is embedded independently (the model averages
    over its input, so chunking is what preserves temporal resolution).

    Args:
        path: Path to the audio file.
        chunk_seconds: Chunk length (default 5.0 s, the model's native
            training length).
        chunk_hop_seconds: Hop between chunk starts (default 2.5 s).
        batch_size: Chunks embedded per forward pass (default 16).
        checkpoint_path: Optional custom checkpoint; default downloads the
            official Zenodo checkpoint on first use.

    Returns:
        A ``(n_chunks, embed_dim)`` float32 array of L2-normalized
        embeddings, in temporal order. Chunk ``i`` starts at
        ``i * chunk_hop_seconds`` seconds.
    """
    import torch

    model = _load_model(checkpoint_path)
    audio = decode_audio(path, sr=NEURAL_SR)

    chunk_len = int(chunk_seconds * NEURAL_SR)
    hop_len = int(chunk_hop_seconds * NEURAL_SR)
    if len(audio) < chunk_len:
        audio = np.pad(audio, (0, chunk_len - len(audio)))

    starts = list(range(0, len(audio) - chunk_len + 1, hop_len))
    chunks = np.stack([audio[s : s + chunk_len] for s in starts])

    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(chunks), batch_size):
            batch = torch.from_numpy(chunks[i : i + batch_size])
            out = model(batch, audio=True)  # (batch, 1, embed_dim)
            embeddings.append(out.squeeze(1).cpu().numpy())

    matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def neural_similarity(
    path_a: str,
    path_b: str,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_hop_seconds: float = DEFAULT_CHUNK_HOP_SECONDS,
    checkpoint_path: str | None = None,
) -> dict:
    """Neural content similarity between two files, in detect()'s NFP format.

    Each chunk of A is compared (cosine) against every chunk of B; a chunk
    "matches" when its best cosine clears ``match_threshold``.

    Returns a dict whose keys are named to plug STRAIGHT into
    :func:`audiotwin.core.detect` / :func:`detect_relation` /
    :func:`combine_scores`::

        nfp = neural_similarity(a, b)
        verdict = detect(a, b, nfp_score=nfp["nfp_score"],
                         nfp_segments_matched=nfp["nfp_segments_matched"],
                         nfp_coverage=nfp["nfp_coverage"])

    Args:
        path_a: Query track path.
        path_b: Reference track path.
        match_threshold: Cosine floor for a chunk to count as matched
            (default 0.60).
        chunk_seconds: See :func:`neural_embedding`.
        chunk_hop_seconds: See :func:`neural_embedding`.
        checkpoint_path: Optional custom checkpoint.

    Returns:
        A dict with ``nfp_score`` (mean over A's chunks of the best cosine
        against B, clamped to [0, 1]), ``nfp_segments_matched`` (number of
        A-chunks above the threshold) and ``nfp_coverage`` (that count over
        A's total chunks).
    """
    emb_a = neural_embedding(
        path_a,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )
    emb_b = neural_embedding(
        path_b,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )

    similarity_matrix = emb_a @ emb_b.T  # (chunks_a, chunks_b), cosine
    best_per_chunk = similarity_matrix.max(axis=1)

    matched = int((best_per_chunk >= match_threshold).sum())
    return {
        "nfp_score": float(np.clip(best_per_chunk.mean(), 0.0, 1.0)),
        "nfp_segments_matched": matched,
        "nfp_coverage": matched / len(best_per_chunk),
    }


def neural_match_points(
    path_a: str,
    path_b: str,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_hop_seconds: float = DEFAULT_CHUNK_HOP_SECONDS,
    checkpoint_path: str | None = None,
) -> list[tuple[float, float, float]]:
    """Chunk-level correspondences in classify_edit()'s match-point format.

    For each chunk of A whose best cosine against B clears the threshold,
    emit ``(t_query, t_ref, score)`` at the two chunks' center times. The
    output feeds :func:`audiotwin.core.classify_edit` /
    :func:`fit_temporal_alignment` directly — a transformation-robust
    alternative to landmark match points (the embeddings survive EQ,
    overdubs and moderate pitch/time edits that break spectral-peak
    hashing)::

        points = neural_match_points(edited.mp3, original.mp3)
        verdict = classify_edit(points, query_duration=..., ref_duration=...)

    Args:
        path_a: Query track path.
        path_b: Reference track path.
        match_threshold: Cosine floor for emitting a point (default 0.60).
        chunk_seconds: See :func:`neural_embedding`.
        chunk_hop_seconds: See :func:`neural_embedding`.
        checkpoint_path: Optional custom checkpoint.

    Returns:
        ``(t_query, t_ref, score)`` triples, one per matched A-chunk,
        sorted by ``t_query``.

    Note:
        Points sit at CHUNK CENTERS, so they can never reach the track
        edges: maximum achievable coverage is
        ``(duration - chunk_seconds) / duration`` (e.g. 0.83 for a 30 s
        track with 5 s chunks). When feeding
        :func:`audiotwin.core.classify_edit`, lower
        ``full_coverage_threshold`` accordingly or use longer tracks.
    """
    emb_a = neural_embedding(
        path_a,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )
    emb_b = neural_embedding(
        path_b,
        chunk_seconds=chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        checkpoint_path=checkpoint_path,
    )

    similarity_matrix = emb_a @ emb_b.T
    center = chunk_seconds / 2.0

    points = []
    for i in range(similarity_matrix.shape[0]):
        j = int(similarity_matrix[i].argmax())
        score = float(similarity_matrix[i, j])
        if score >= match_threshold:
            points.append(
                (
                    i * chunk_hop_seconds + center,
                    j * chunk_hop_seconds + center,
                    score,
                )
            )
    return points
