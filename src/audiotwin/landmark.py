"""Spectral-peak (landmark) fingerprinting — Wang 2003, Shazam-style.

Robust to noise and additive mixing, and returns the exact temporal
position of each correspondence. This is the detector behind SAMPLE
(a reused fragment), MASHUP (multiple fragments from distinct sources),
and supporting evidence for REMIX.

Requires the ``[landmark]`` extra (scipy). All heavy imports are lazy so
the core install stays light.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import numpy as np

from audiotwin.audio import decode_audio
from audiotwin.core import compute_coverage

#: Default landmark extraction sample rate.
DEFAULT_LANDMARK_SR = 11025

#: Aligned-hash count at which sample/mashup confidence saturates to 1.0.
#: ~50 aligned hashes is already overwhelming evidence for a real match;
#: beyond that, more hashes shouldn't inflate confidence further.
CONFIDENCE_SATURATION_HASHES = 50

# Hash bit layout (32 bits total, documented for reproducibility):
#   bits 23..31 (9 bits): anchor peak frequency bin   (0..511)
#   bits 14..22 (9 bits): target peak frequency bin   (0..511)
#   bits  0..13 (14 bits): Δt in STFT frames          (0..16383)
# With n_fft=512 there are 257 frequency bins, fitting comfortably in
# 9 bits; Δt at the default hop (256 samples @ 11025 Hz ≈ 23 ms/frame)
# covers up to ~380 s, far beyond the target zone's Δt_max.
_FREQ_BITS = 9
_DT_BITS = 14


def _require_scipy():
    try:
        import scipy.ndimage
        import scipy.signal
    except ImportError as exc:
        raise ImportError(
            "the landmark module requires scipy — install it with 'pip install audiotwin[landmark]'"
        ) from exc
    return scipy.signal, scipy.ndimage


def _spectral_whitening(
    magnitude: np.ndarray,
    envelope_bins: int,
    floor: float,
) -> np.ndarray:
    """Divide each spectrogram column by its smoothed spectral envelope.

    The envelope is a uniform (moving-average) filter along the FREQUENCY
    axis, so the division flattens spectral roll-off — high-frequency peaks
    compete fairly with the low-frequency energy mass — while preserving
    the LOCAL contrast that makes a peak a peak. (A running-peak AGC in the
    style of Stowell & Plumbley 2007 was tried first and rejected: it
    saturates most bins to exactly 1.0, ties collapse the top-K peak
    selection into lattice artifacts, and specificity is destroyed — white
    noise matched indexed tracks with hundreds of aligned hashes.)
    """
    _, scipy_ndimage = _require_scipy()
    envelope = scipy_ndimage.uniform_filter1d(magnitude, size=envelope_bins, axis=0)
    return magnitude / (envelope + floor)


def extract_landmarks(
    audio: np.ndarray,
    sr: int = DEFAULT_LANDMARK_SR,
    n_fft: int = 512,
    hop_length: int = 256,
    peak_neighborhood: tuple[int, int] = (30, 30),
    target_density: float = 25.0,
    fan_out: int = 5,
    target_zone: tuple[float, float, int, int] = (0.5, 3.0, -32, 32),
    whiten: bool = True,
    whitening_envelope_bins: int = 31,
    whitening_floor: float = 1e-6,
) -> list[tuple[int, float]]:
    """Extract Wang-2003 landmark hashes from mono PCM audio.

    Pipeline: STFT magnitude → optional adaptive whitening →
    non-maximum-suppression peak picking with an adaptive threshold
    targeting ``target_density`` peaks/second → anchor/target pairing
    within ``target_zone`` → 32-bit hash packing (see the bit-layout
    comment at module top).

    Args:
        audio: Mono float32 PCM (output of :func:`audiotwin.audio.decode_audio`).
        sr: Sample rate of ``audio`` (default 11025).
        n_fft: STFT window size (default 512).
        hop_length: STFT hop in samples (default 256).
        peak_neighborhood: ``(freq_bins, time_frames)`` size of the
            non-maximum-suppression zone around each peak (default (30, 30)).
        target_density: Approximate peaks per second to keep (default 25).
        fan_out: Number of target peaks paired with each anchor (default 5).
        target_zone: ``(dt_min_s, dt_max_s, dfreq_min_bins, dfreq_max_bins)``
            window in which to look for target peaks (default
            (0.5, 3.0, -32, 32)).
        whiten: Apply spectral whitening before peak picking (default
            True): each column is divided by its smoothed spectral envelope
            so high-frequency peaks compete fairly with the low-frequency
            energy mass. **Changing this changes the extracted hashes** —
            use the same value at index and query time.
        whitening_envelope_bins: Frequency width (bins) of the envelope's
            moving-average filter (default 31).
        whitening_floor: Envelope floor preventing noise amplification in
            silent bins (default 1e-6).

    Returns:
        A list of ``(hash_int, t_anchor_seconds)`` tuples.
    """
    scipy_signal, scipy_ndimage = _require_scipy()

    if len(audio) < n_fft:
        return []

    _, _, stft = scipy_signal.stft(
        audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, padded=False
    )
    magnitude = np.abs(stft)
    duration = len(audio) / sr

    if whiten:
        magnitude = _spectral_whitening(
            magnitude,
            envelope_bins=whitening_envelope_bins,
            floor=whitening_floor,
        )

    # Non-maximum suppression: a bin survives when it is the maximum of its
    # neighborhood. The adaptive part is the global top-K selection below.
    local_max = magnitude == scipy_ndimage.maximum_filter(magnitude, size=peak_neighborhood)
    local_max &= magnitude > 0.0

    freqs, frames = np.nonzero(local_max)
    if len(freqs) == 0:
        return []

    max_peaks = max(1, int(round(target_density * duration)))
    if len(freqs) > max_peaks:
        strengths = magnitude[freqs, frames]
        keep = np.argsort(strengths)[::-1][:max_peaks]
        freqs, frames = freqs[keep], frames[keep]

    # Sort peaks by time for anchor→target pairing.
    order = np.argsort(frames, kind="stable")
    freqs, frames = freqs[order], frames[order]

    dt_min_s, dt_max_s, df_min, df_max = target_zone
    frames_per_second = sr / hop_length
    dt_min = max(1, int(round(dt_min_s * frames_per_second)))
    dt_max = int(round(dt_max_s * frames_per_second))

    landmarks: list[tuple[int, float]] = []
    n_peaks = len(frames)
    for i in range(n_peaks):
        f1, t1 = int(freqs[i]), int(frames[i])
        # Peaks are time-sorted; scan forward within the Δt window.
        paired = 0
        j = i + 1
        while j < n_peaks and paired < fan_out:
            dt = int(frames[j]) - t1
            if dt > dt_max:
                break
            if dt >= dt_min:
                df = int(freqs[j]) - f1
                if df_min <= df <= df_max:
                    f2 = int(freqs[j])
                    h = (f1 << (_FREQ_BITS + _DT_BITS)) | (f2 << _DT_BITS) | dt
                    landmarks.append((h, t1 / frames_per_second))
                    paired += 1
            j += 1

    return landmarks


def _match_landmarks(
    query_landmarks: list[tuple[int, float]],
    conn: sqlite3.Connection,
    min_aligned_hashes: int,
    offset_bin_width: float,
) -> list[dict]:
    """Match extracted query landmarks against the index.

    For each candidate track, histogram the ``t_ref - t_query`` offsets;
    a histogram peak of at least ``min_aligned_hashes`` consistent offsets
    is a match (Wang 2003's time-coherence test). Adjacent histogram bins
    are fused pairwise: a true offset landing near a bin boundary splits
    its votes between two neighboring bins, so the peak is searched over
    ``bin ∪ bin+1`` and the reported offset is the mean of the fused
    pairs' actual offsets (more accurate than the bin center).
    """
    if not query_landmarks:
        return []

    by_hash: dict[int, list[float]] = defaultdict(list)
    for h, t_query in query_landmarks:
        by_hash[h].append(t_query)

    # (track_id, offset_bin) -> list of (t_query, t_ref)
    pair_bins: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)

    hashes = list(by_hash.keys())
    chunk_size = 500  # stay under SQLite's default 999-variable limit
    for start in range(0, len(hashes), chunk_size):
        chunk = hashes[start : start + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT hash, track_id, t_anchor FROM landmarks WHERE hash IN ({placeholders})",
            chunk,
        )
        for h, track_id, t_ref in rows:
            for t_query in by_hash[h]:
                offset_bin = int(round((t_ref - t_query) / offset_bin_width))
                pair_bins[(track_id, offset_bin)].append((t_query, t_ref))

    # Regroup bins per track, then find the best FUSED (bin, bin+1) window.
    bins_per_track: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(dict)
    for (track_id, offset_bin), pairs in pair_bins.items():
        bins_per_track[track_id][offset_bin] = pairs

    best_per_track: dict[str, list[tuple[float, float]]] = {}
    for track_id, bins in bins_per_track.items():
        best_pairs: list[tuple[float, float]] = []
        for offset_bin, pairs in bins.items():
            fused = pairs + bins.get(offset_bin + 1, [])
            if len(fused) > len(best_pairs):
                best_pairs = fused
        best_per_track[track_id] = best_pairs

    results = []
    n_query_hashes = len(query_landmarks)
    for track_id, pairs in best_per_track.items():
        if len(pairs) < min_aligned_hashes:
            continue
        offset_seconds = sum(t_r - t_q for t_q, t_r in pairs) / len(pairs)
        results.append(
            {
                "track_id": track_id,
                "offset_seconds": offset_seconds,
                "aligned_hashes": len(pairs),
                "query_hashes": n_query_hashes,
                "score": min(1.0, len(pairs) / n_query_hashes),
                "pitch_shift_semitones": 0,
                "match_points": [(t_q, t_r, 1.0) for t_q, t_r in sorted(pairs)],
            }
        )

    results.sort(key=lambda r: r["aligned_hashes"], reverse=True)
    return results


class LandmarkIndex:
    """SQLite-backed landmark index (stdlib only, ``":memory:"`` accepted).

    Schema: ``landmarks(hash INTEGER, track_id TEXT, t_anchor REAL)`` with
    an index on ``hash``, plus a ``tracks(track_id TEXT PRIMARY KEY,
    n_landmarks INTEGER)`` bookkeeping table. Insertions are batched with
    ``executemany``.
    """

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS landmarks (
                hash INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                t_anchor REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_landmarks_hash ON landmarks(hash);
            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                n_landmarks INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()

    def add_track(self, track_id: str, audio_path: str, **extract_params) -> int:
        """Decode, extract landmarks, and insert them under ``track_id``.

        Raises:
            ValueError: If ``track_id`` is already indexed (no silent
                overwrite — call :meth:`remove_track` first).

        Returns:
            The number of landmarks added.
        """
        exists = self._conn.execute(
            "SELECT 1 FROM tracks WHERE track_id = ?", (track_id,)
        ).fetchone()
        if exists:
            raise ValueError(
                f"track_id {track_id!r} is already indexed; call "
                "remove_track() first to replace it."
            )

        sr = extract_params.pop("sr", DEFAULT_LANDMARK_SR)
        audio = decode_audio(audio_path, sr=sr)
        landmarks = extract_landmarks(audio, sr=sr, **extract_params)

        self._conn.executemany(
            "INSERT INTO landmarks (hash, track_id, t_anchor) VALUES (?, ?, ?)",
            [(h, track_id, t) for h, t in landmarks],
        )
        self._conn.execute(
            "INSERT INTO tracks (track_id, n_landmarks) VALUES (?, ?)",
            (track_id, len(landmarks)),
        )
        self._conn.commit()
        return len(landmarks)

    def add_track_landmarks(
        self, track_id: str, landmarks: list[tuple[int, float]]
    ) -> int:
        """Insert PRE-EXTRACTED landmarks under ``track_id``.

        The cached counterpart of :meth:`add_track` for callers that store
        :func:`extract_landmarks` output per track (e.g. a large-catalog
        pipeline) and don't want audio re-decoded per pair.

        Raises:
            ValueError: If ``track_id`` is already indexed.

        Returns:
            The number of landmarks added.
        """
        exists = self._conn.execute(
            "SELECT 1 FROM tracks WHERE track_id = ?", (track_id,)
        ).fetchone()
        if exists:
            raise ValueError(
                f"track_id {track_id!r} is already indexed; call "
                "remove_track() first to replace it."
            )
        self._conn.executemany(
            "INSERT INTO landmarks (hash, track_id, t_anchor) VALUES (?, ?, ?)",
            [(int(h), track_id, float(t)) for h, t in landmarks],
        )
        self._conn.execute(
            "INSERT INTO tracks (track_id, n_landmarks) VALUES (?, ?)",
            (track_id, len(landmarks)),
        )
        self._conn.commit()
        return len(landmarks)

    def query_landmarks(
        self,
        landmarks: list[tuple[int, float]],
        min_aligned_hashes: int = 10,
        offset_bin_width: float = 0.2,
    ) -> list[dict]:
        """Match PRE-EXTRACTED query landmarks against the index.

        The cached counterpart of :meth:`query` (without the decode /
        extract / pitch-shift stages): given the same landmark lists,
        returns exactly what :meth:`query` would.
        """
        return _match_landmarks(
            landmarks, self._conn, min_aligned_hashes, offset_bin_width
        )

    def remove_track(self, track_id: str) -> int:
        """Remove a track's landmarks. Returns the number of rows deleted."""
        cursor = self._conn.execute("DELETE FROM landmarks WHERE track_id = ?", (track_id,))
        self._conn.execute("DELETE FROM tracks WHERE track_id = ?", (track_id,))
        self._conn.commit()
        return cursor.rowcount

    def track_ids(self) -> list[str]:
        """List indexed track ids."""
        rows = self._conn.execute("SELECT track_id FROM tracks ORDER BY track_id")
        return [r[0] for r in rows]

    def query(
        self,
        audio_path: str,
        min_aligned_hashes: int = 10,
        offset_bin_width: float = 0.2,
        pitch_shift_range: int = 0,
        **extract_params,
    ) -> list[dict]:
        """Match a query file against the index.

        Args:
            audio_path: Path to the query audio file.
            min_aligned_hashes: Minimum offset-consistent hashes for a
                match to be reported (default 10).
            offset_bin_width: Width (seconds) of the offset-histogram bins
                (default 0.2).
            pitch_shift_range: 0 disables (default); N also queries the
                query audio pitch-shifted by ±1..±N semitones and keeps the
                best result per track. Requires librosa (the ``[cover]`` or
                ``[all]`` extra).
            **extract_params: Forwarded to :func:`extract_landmarks`.

        Returns:
            Matches sorted by ``aligned_hashes`` (desc); see the module
            README for the per-match dict schema. ``match_points`` uses the
            same ``(t_query, t_ref, score)`` format that
            :func:`audiotwin.core.classify_edit` and
            :func:`audiotwin.core.fit_temporal_alignment` accept.
        """
        sr = extract_params.pop("sr", DEFAULT_LANDMARK_SR)
        audio = decode_audio(audio_path, sr=sr)

        landmarks = extract_landmarks(audio, sr=sr, **extract_params)
        results = _match_landmarks(landmarks, self._conn, min_aligned_hashes, offset_bin_width)

        if pitch_shift_range > 0:
            try:
                import librosa
            except ImportError as exc:
                raise ImportError(
                    "pitch_shift_range requires librosa — install it with "
                    "'pip install audiotwin[cover]' (or audiotwin[all])"
                ) from exc

            best_by_track: dict[str, dict] = {r["track_id"]: r for r in results}
            shifts = [s for s in range(-pitch_shift_range, pitch_shift_range + 1) if s != 0]
            for shift in shifts:
                shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
                shifted_landmarks = extract_landmarks(shifted, sr=sr, **extract_params)
                for r in _match_landmarks(
                    shifted_landmarks, self._conn, min_aligned_hashes, offset_bin_width
                ):
                    r["pitch_shift_semitones"] = shift
                    current = best_by_track.get(r["track_id"])
                    if current is None or r["aligned_hashes"] > current["aligned_hashes"]:
                        best_by_track[r["track_id"]] = r
            results = sorted(
                best_by_track.values(), key=lambda r: r["aligned_hashes"], reverse=True
            )

        return results

    def close(self) -> None:
        self._conn.close()


def _saturating_confidence(aligned_hashes: int) -> float:
    """Confidence proportional to evidence, saturating at
    :data:`CONFIDENCE_SATURATION_HASHES` aligned hashes."""
    return min(1.0, aligned_hashes / CONFIDENCE_SATURATION_HASHES)


def classify_sample(
    query_result: dict,
    query_duration: float,
    ref_duration: float,
    localized_max_coverage: float = 0.35,
) -> dict:
    """Decide whether a landmark match is a LOCALIZED sample.

    Note:
        Cette fonction APPLIQUE des seuils et rend un verdict. Les
        pipelines qui construisent leur propre couche de décision (ML ou
        règles maison) doivent utiliser
        :func:`audiotwin.scores.extract_all_scores`, qui expose les
        signaux bruts sans aucune décision.


    Reuses :func:`audiotwin.core.compute_coverage` on the match points: a
    match spanning less than ``localized_max_coverage`` of the query is a
    localized fragment (SAMPLE territory); anything wider is a global
    relation (edit/remix/duplicate) and not flagged here.

    Args:
        query_result: One element of :meth:`LandmarkIndex.query`'s output.
        query_duration: Query track duration in seconds.
        ref_duration: Reference track duration in seconds.
        localized_max_coverage: Query-coverage ceiling for the match to
            still count as localized (default 0.35).

    Returns:
        A dict with ``is_localized_match``, the sample's temporal bounds on
        both tracks (``None`` when not localized), ``aligned_hashes``,
        both coverages and ``confidence`` (saturating in aligned hashes,
        see :data:`CONFIDENCE_SATURATION_HASHES`; 0.0 when not localized).
    """
    match_points = query_result["match_points"]
    coverage = compute_coverage(
        match_points,
        list(range(len(match_points))),
        query_duration,
        ref_duration,
    )

    is_localized = bool(match_points) and coverage["coverage_query"] <= localized_max_coverage

    if is_localized:
        t_queries = [p[0] for p in match_points]
        t_refs = [p[1] for p in match_points]
        bounds = {
            "sample_start_query": min(t_queries),
            "sample_end_query": max(t_queries),
            "sample_start_ref": min(t_refs),
            "sample_end_ref": max(t_refs),
        }
        confidence = _saturating_confidence(query_result["aligned_hashes"])
    else:
        bounds = {
            "sample_start_query": None,
            "sample_end_query": None,
            "sample_start_ref": None,
            "sample_end_ref": None,
        }
        confidence = 0.0

    return {
        "is_localized_match": is_localized,
        **bounds,
        "aligned_hashes": query_result["aligned_hashes"],
        "coverage_query": coverage["coverage_query"],
        "coverage_ref": coverage["coverage_ref"],
        "confidence": confidence,
    }


def classify_mashup(
    query_results: list[dict],
    query_duration: float,
    min_sources: int = 2,
    min_aligned_hashes_per_source: int = 15,
) -> dict:
    """Detect a MASHUP pattern: strong matches against several distinct
    tracks covering mostly disjoint regions of the query.

    Note:
        Cette fonction APPLIQUE des seuils et rend un verdict. Les
        pipelines qui construisent leur propre couche de décision (ML ou
        règles maison) doivent utiliser
        :func:`audiotwin.scores.extract_all_scores`, qui expose les
        signaux bruts sans aucune décision.


    Sources are accepted greedily by descending ``aligned_hashes``; a
    candidate is rejected when its query-time region overlaps an already
    accepted region by ≥ 30% (of the shorter region). Region bounds use
    the 5th–95th percentiles of the match points' query times rather than
    min/max: a handful of stray hash collisions must not stretch a
    region into its neighbors and veto a genuine mashup source.

    Args:
        query_results: Full output of :meth:`LandmarkIndex.query`.
        query_duration: Query track duration in seconds.
        min_sources: Minimum distinct sources (default 2).
        min_aligned_hashes_per_source: Evidence floor per source
            (default 15).

    Returns:
        A dict with ``is_mashup_pattern``, ``source_count``, ``sources``
        (track_id + region bounds + aligned_hashes), ``coverage_total``
        (share of the query covered by the union of regions) and
        ``confidence`` (mean per-source saturating confidence ×
        coverage_total — both more evidence per source and more of the
        query explained increase it).
    """
    candidates = [r for r in query_results if r["aligned_hashes"] >= min_aligned_hashes_per_source]
    candidates.sort(key=lambda r: r["aligned_hashes"], reverse=True)

    accepted: list[dict] = []
    regions: list[tuple[float, float]] = []
    seen_tracks: set[str] = set()

    for r in candidates:
        if r["track_id"] in seen_tracks or not r["match_points"]:
            continue
        t_queries = np.asarray([p[0] for p in r["match_points"]])
        start, end = (float(x) for x in np.percentile(t_queries, [5.0, 95.0]))
        length = max(end - start, 1e-9)

        overlaps_too_much = False
        for a_start, a_end in regions:
            overlap = max(0.0, min(end, a_end) - max(start, a_start))
            if overlap / min(length, max(a_end - a_start, 1e-9)) >= 0.30:
                overlaps_too_much = True
                break
        if overlaps_too_much:
            continue

        accepted.append(
            {
                "track_id": r["track_id"],
                "region_start": start,
                "region_end": end,
                "aligned_hashes": r["aligned_hashes"],
            }
        )
        regions.append((start, end))
        seen_tracks.add(r["track_id"])

    # Union of accepted regions over the query timeline.
    coverage_total = 0.0
    if regions:
        sorted_regions = sorted(regions)
        union = 0.0
        current_start, current_end = sorted_regions[0]
        for start, end in sorted_regions[1:]:
            if start > current_end:
                union += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        union += current_end - current_start
        coverage_total = min(1.0, union / query_duration)

    is_mashup = len(accepted) >= min_sources

    if is_mashup:
        mean_conf = sum(_saturating_confidence(s["aligned_hashes"]) for s in accepted) / len(
            accepted
        )
        confidence = min(1.0, mean_conf * coverage_total)
    else:
        confidence = 0.0

    return {
        "is_mashup_pattern": is_mashup,
        "source_count": len(accepted),
        "sources": accepted,
        "coverage_total": coverage_total,
        "confidence": confidence,
    }
