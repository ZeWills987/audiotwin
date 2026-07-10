"""audiotwin — lightweight detection of audio twins (same master, any encoding).

The library takes two audio files and returns raw similarity scores. It makes
no business decisions (dedup merging, canonical selection, ...) — those stay
with the caller.
"""

from audiotwin.core import (
    DEFAULT_CHROMAPRINT_THRESHOLD,
    DEFAULT_NFP_THRESHOLD,
    AudioTooShortError,
    combine_scores,
    compare_fingerprints,
    compute_fingerprint,
    detect,
    file_hash,
)

__version__ = "0.1.0"

__all__ = [
    "AudioTooShortError",
    "DEFAULT_CHROMAPRINT_THRESHOLD",
    "DEFAULT_NFP_THRESHOLD",
    "combine_scores",
    "compare_fingerprints",
    "compute_fingerprint",
    "detect",
    "file_hash",
]
