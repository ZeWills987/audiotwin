"""Tests for suggest_relation — the rule-based convenience aggregator."""

import pytest

from audiotwin import suggest_relation


def _relations(result):
    return [h["relation"] for h in result["hypotheses"]]


def test_no_inputs_yields_no_relation():
    result = suggest_relation()
    assert _relations(result) == ["NO_RELATION"]
    assert result["hypotheses"][0]["confidence"] == 0.0


def test_nothing_clears_threshold_yields_no_relation():
    result = suggest_relation(
        chromaprint_score=0.30,
        cover_result={"similarity": 0.20, "transposition_semitones": 0},
        edit_result={"edit_type_hint": "no_relation", "confidence": 0.0},
        sample_result={"is_localized_match": False, "confidence": 0.0},
        mashup_result={"is_mashup_pattern": False, "confidence": 0.0},
        instrumental_result={"is_instrumental_pair": False, "confidence": 0.0},
    )
    assert _relations(result) == ["NO_RELATION"]


def test_duplicate_from_chromaprint():
    result = suggest_relation(chromaprint_score=0.95)
    assert _relations(result) == ["DUPLICATE"]
    assert result["hypotheses"][0]["confidence"] == pytest.approx(0.95 * 0.9)


def test_remaster_from_scores():
    result = suggest_relation(chromaprint_score=0.70, nfp_score=0.95)
    assert _relations(result) == ["REMASTER"]
    assert result["hypotheses"][0]["confidence"] == pytest.approx(0.95 * 0.9)


def test_priority_order_full_stack():
    # Everything fires at once: order must follow the documented priority,
    # NOT the confidences.
    result = suggest_relation(
        chromaprint_score=0.95,
        edit_result={
            "edit_type_hint": "trim_or_extend",
            "confidence": 0.99,
            "slope": 1.0,
            "coverage_query": 1.0,
            "coverage_ref": 0.5,
        },
        sample_result={
            "is_localized_match": True,
            "confidence": 0.6,
            "aligned_hashes": 30,
            "coverage_query": 0.2,
        },
        mashup_result={
            "is_mashup_pattern": True,
            "confidence": 0.5,
            "source_count": 3,
            "coverage_total": 0.9,
        },
        cover_result={"similarity": 0.99, "transposition_semitones": 2},
        instrumental_result={
            "is_instrumental_pair": True,
            "confidence": 0.4,
            "vocal_track": "a",
            "vocal_gap": 0.6,
        },
    )
    assert _relations(result) == [
        "DUPLICATE",
        "MASHUP",
        "SAMPLE",
        "EDIT",
        "INSTRUMENTAL",
        "COVER",
    ]


def test_confidences_carried_from_sources():
    result = suggest_relation(
        sample_result={
            "is_localized_match": True,
            "confidence": 0.62,
            "aligned_hashes": 31,
            "coverage_query": 0.2,
        },
        cover_result={"similarity": 0.91, "transposition_semitones": 5},
    )
    by_relation = {h["relation"]: h for h in result["hypotheses"]}
    assert by_relation["SAMPLE"]["confidence"] == 0.62
    assert by_relation["COVER"]["confidence"] == 0.91


def test_edit_full_match_is_not_an_edit_hypothesis():
    result = suggest_relation(
        edit_result={
            "edit_type_hint": "full_match",
            "confidence": 0.9,
            "slope": 1.0,
            "coverage_query": 1.0,
            "coverage_ref": 1.0,
        }
    )
    assert _relations(result) == ["NO_RELATION"]


def test_custom_cover_threshold():
    cover = {"similarity": 0.55, "transposition_semitones": 0}
    assert _relations(suggest_relation(cover_result=cover)) == ["NO_RELATION"]
    result = suggest_relation(cover_result=cover, cover_similarity_threshold=0.50)
    assert _relations(result) == ["COVER"]


def test_evidence_strings_present():
    result = suggest_relation(chromaprint_score=0.95, nfp_score=0.97)
    hypothesis = result["hypotheses"][0]
    assert "chromaprint_score=0.950" in hypothesis["evidence"]
    assert "nfp_score=0.970" in hypothesis["evidence"]
