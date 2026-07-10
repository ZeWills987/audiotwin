"""Command-line interface for audiotwin.

Examples:
    audiotwin compare a.mp3 b.mp3 --json
    audiotwin fingerprint track.mp3
"""

from __future__ import annotations

import argparse
import json
import sys

from audiotwin.core import (
    DEFAULT_CHROMAPRINT_THRESHOLD,
    DEFAULT_NFP_THRESHOLD,
    AudioTooShortError,
    compute_fingerprint,
    detect,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiotwin",
        description="Detect audio twins — two files that are the same master.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="Compare two audio files.")
    compare.add_argument("file_a", help="First audio file.")
    compare.add_argument("file_b", help="Second audio file.")
    compare.add_argument("--json", action="store_true", help="Emit JSON output.")
    compare.add_argument(
        "--max-duration",
        type=int,
        default=120,
        help="Seconds of leading audio to fingerprint (default: 120).",
    )
    compare.add_argument("--nfp-score", type=float, default=None, help="Optional NFP score.")
    compare.add_argument(
        "--nfp-segments-matched", type=int, default=None, help="Optional NFP metadata."
    )
    compare.add_argument("--nfp-coverage", type=float, default=None, help="Optional NFP metadata.")
    compare.add_argument(
        "--chromaprint-threshold",
        type=float,
        default=DEFAULT_CHROMAPRINT_THRESHOLD,
        help=f"Chromaprint match threshold (default: {DEFAULT_CHROMAPRINT_THRESHOLD}).",
    )
    compare.add_argument(
        "--nfp-threshold",
        type=float,
        default=DEFAULT_NFP_THRESHOLD,
        help=f"NFP confirmation threshold (default: {DEFAULT_NFP_THRESHOLD}).",
    )

    fingerprint = sub.add_parser("fingerprint", help="Print a file's Chromaprint fingerprint.")
    fingerprint.add_argument("file", help="Audio file.")
    fingerprint.add_argument(
        "--max-duration",
        type=int,
        default=120,
        help="Seconds of leading audio to fingerprint (default: 120).",
    )

    return parser


def _print_human(result: dict) -> None:
    verdict = "DUPLICATE" if result["is_duplicate"] else "distinct"
    print(f"{result['track_a']}")
    print(f"{result['track_b']}")
    print(f"  verdict          : {verdict}")
    print(f"  confidence       : {result['confidence']:.3f}")
    print(f"  file hash match  : {result['file_hash_match']}")
    print(f"  chromaprint score: {result['chromaprint_score']:.3f}")
    if result["nfp_score"] is not None:
        print(f"  nfp score        : {result['nfp_score']:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "compare":
            result = detect(
                args.file_a,
                args.file_b,
                nfp_score=args.nfp_score,
                nfp_segments_matched=args.nfp_segments_matched,
                nfp_coverage=args.nfp_coverage,
                max_duration=args.max_duration,
                chromaprint_threshold=args.chromaprint_threshold,
                nfp_threshold=args.nfp_threshold,
            )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                _print_human(result)
            return 0

        if args.command == "fingerprint":
            print(compute_fingerprint(args.file, max_duration=args.max_duration))
            return 0
    except AudioTooShortError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
