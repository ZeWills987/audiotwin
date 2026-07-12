"""Command-line interface for audiotwin.

Examples:
    audiotwin compare a.mp3 b.mp3 --json
    audiotwin classify a.mp3 b.mp3 --json
    audiotwin fingerprint track.mp3
"""

from __future__ import annotations

import argparse
import json
import sys

from audiotwin.core import (
    DEFAULT_CHROMAPRINT_THRESHOLD,
    DEFAULT_FULL_COVERAGE_THRESHOLD,
    DEFAULT_MIN_INLIERS,
    DEFAULT_NFP_THRESHOLD,
    DEFAULT_REMASTER_CHROMAPRINT_MIN,
    DEFAULT_REMASTER_NFP_THRESHOLD,
    DEFAULT_RESIDUAL_THRESHOLD,
    DEFAULT_SPEED_CHANGE_EPSILON,
    AudioTooShortError,
    classify_edit,
    compute_fingerprint,
    detect,
    detect_relation,
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

    classify = sub.add_parser(
        "classify", help="Classify two audio files as DUPLICATE, REMASTER, or unrelated."
    )
    classify.add_argument("file_a", help="First audio file.")
    classify.add_argument("file_b", help="Second audio file.")
    classify.add_argument("--json", action="store_true", help="Emit JSON output.")
    classify.add_argument(
        "--max-duration",
        type=int,
        default=120,
        help="Seconds of leading audio to fingerprint (default: 120).",
    )
    classify.add_argument("--nfp-score", type=float, default=None, help="Optional NFP score.")
    classify.add_argument(
        "--nfp-segments-matched", type=int, default=None, help="Optional NFP metadata."
    )
    classify.add_argument(
        "--nfp-coverage", type=float, default=None, help="Optional NFP metadata."
    )
    classify.add_argument(
        "--duplicate-threshold",
        type=float,
        default=DEFAULT_CHROMAPRINT_THRESHOLD,
        help=f"Chromaprint match threshold (default: {DEFAULT_CHROMAPRINT_THRESHOLD}).",
    )
    classify.add_argument(
        "--remaster-chromaprint-min",
        type=float,
        default=DEFAULT_REMASTER_CHROMAPRINT_MIN,
        help=f"Lower Chromaprint bound for REMASTER (default: {DEFAULT_REMASTER_CHROMAPRINT_MIN}).",
    )
    classify.add_argument(
        "--remaster-nfp-threshold",
        type=float,
        default=DEFAULT_REMASTER_NFP_THRESHOLD,
        help=(
            "NFP confirmation threshold for REMASTER "
            f"(default: {DEFAULT_REMASTER_NFP_THRESHOLD})."
        ),
    )

    classify_edit_p = sub.add_parser(
        "classify-edit",
        help="Classify an edit relation (speed change, trim/extend) from match points.",
    )
    classify_edit_p.add_argument(
        "matches_file",
        help="JSON file containing a list of [t_query, t_ref, score] triples.",
    )
    classify_edit_p.add_argument(
        "--query-duration", type=float, required=True, help="Query track duration (s)."
    )
    classify_edit_p.add_argument(
        "--ref-duration", type=float, required=True, help="Reference track duration (s)."
    )
    classify_edit_p.add_argument("--json", action="store_true", help="Emit JSON output.")
    classify_edit_p.add_argument(
        "--min-inliers",
        type=int,
        default=DEFAULT_MIN_INLIERS,
        help=f"Minimum inliers for a trustworthy fit (default: {DEFAULT_MIN_INLIERS}).",
    )
    classify_edit_p.add_argument(
        "--residual-threshold",
        type=float,
        default=DEFAULT_RESIDUAL_THRESHOLD,
        help=f"Inlier tolerance in seconds (default: {DEFAULT_RESIDUAL_THRESHOLD}).",
    )
    classify_edit_p.add_argument(
        "--speed-change-epsilon",
        type=float,
        default=DEFAULT_SPEED_CHANGE_EPSILON,
        help=(
            "Slope deviation from 1.0 treated as a speed change "
            f"(default: {DEFAULT_SPEED_CHANGE_EPSILON})."
        ),
    )
    classify_edit_p.add_argument(
        "--full-coverage-threshold",
        type=float,
        default=DEFAULT_FULL_COVERAGE_THRESHOLD,
        help=(
            "Coverage above which a side counts as fully covered "
            f"(default: {DEFAULT_FULL_COVERAGE_THRESHOLD})."
        ),
    )
    classify_edit_p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for reproducible RANSAC sampling.",
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


def _print_human_relation(result: dict) -> None:
    print(f"{result['track_a']}")
    print(f"{result['track_b']}")
    print(f"  relation         : {result['relation_type']}")
    print(f"  confidence       : {result['confidence']:.3f}")
    print(f"  file hash match  : {result['file_hash_match']}")
    print(f"  chromaprint score: {result['chromaprint_score']:.3f}")
    if result["nfp_score"] is not None:
        print(f"  nfp score        : {result['nfp_score']:.3f}")
    if result["relation_type"] == "REMASTER" and result["score_gap"] is not None:
        print(f"  score gap (nfp - chromaprint): {result['score_gap']:.3f}")


def _print_human_edit(result: dict) -> None:
    print(f"  edit type hint   : {result['edit_type_hint']}")
    print(f"  confidence       : {result['confidence']:.3f}")
    print(f"  slope (speed)    : {result['slope']:.4f}")
    print(f"  intercept (s)    : {result['intercept']:.3f}")
    print(f"  inliers/outliers : {result['inlier_count']}/{result['outlier_count']}")
    print(f"  coverage query   : {result['coverage_query']:.3f}")
    print(f"  coverage ref     : {result['coverage_ref']:.3f}")
    print(f"  consecutive      : {result['is_consecutive']}")


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

        if args.command == "classify":
            result = detect_relation(
                args.file_a,
                args.file_b,
                nfp_score=args.nfp_score,
                nfp_segments_matched=args.nfp_segments_matched,
                nfp_coverage=args.nfp_coverage,
                max_duration=args.max_duration,
                duplicate_threshold=args.duplicate_threshold,
                remaster_chromaprint_min=args.remaster_chromaprint_min,
                remaster_nfp_threshold=args.remaster_nfp_threshold,
            )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                _print_human_relation(result)
            return 0

        if args.command == "classify-edit":
            with open(args.matches_file, encoding="utf-8") as f:
                raw_matches = json.load(f)
            matches = [tuple(m) for m in raw_matches]
            result = classify_edit(
                matches,
                query_duration=args.query_duration,
                ref_duration=args.ref_duration,
                min_inliers=args.min_inliers,
                residual_threshold=args.residual_threshold,
                speed_change_epsilon=args.speed_change_epsilon,
                full_coverage_threshold=args.full_coverage_threshold,
                random_seed=args.random_seed,
            )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(args.matches_file)
                _print_human_edit(result)
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
