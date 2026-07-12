"""audiotwin — lightweight detection of audio twins (same master, any encoding).

The library takes two audio files and returns raw similarity scores. It makes
no business decisions (dedup merging, canonical selection, ...) — those stay
with the caller.
"""

from audiotwin.core import (
    DEFAULT_CHROMAPRINT_THRESHOLD,
    DEFAULT_NFP_THRESHOLD,
    DEFAULT_REMASTER_CHROMAPRINT_MIN,
    DEFAULT_REMASTER_NFP_THRESHOLD,
    AudioTooShortError,
    classify_edit,
    classify_instrumental_pair,
    classify_relation,
    combine_scores,
    compare_fingerprints,
    compute_coverage,
    compute_fingerprint,
    detect,
    detect_relation,
    file_hash,
    fit_temporal_alignment,
)

__version__ = "0.1.0"

__all__ = [
    "AudioTooShortError",
    "DEFAULT_CHROMAPRINT_THRESHOLD",
    "DEFAULT_NFP_THRESHOLD",
    "DEFAULT_REMASTER_CHROMAPRINT_MIN",
    "DEFAULT_REMASTER_NFP_THRESHOLD",
    "classify_edit",
    "classify_instrumental_pair",
    "classify_relation",
    "combine_scores",
    "compare_fingerprints",
    "compute_coverage",
    "compute_fingerprint",
    "detect",
    "detect_relation",
    "file_hash",
    "fit_temporal_alignment",
]
