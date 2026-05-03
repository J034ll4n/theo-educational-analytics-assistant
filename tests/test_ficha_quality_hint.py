"""Heurística de lacuna na ficha (sem LangChain)."""

from __future__ import annotations

from passos_magico.llm.ficha_quality_hint import (
    ficha_quality_snapshot_extra_lines,
    ficha_suspect_missing_indicators,
    is_zero_pedagogy_score,
)


def test_suspect_when_ida_and_ieg_zero_with_plausible_inde() -> None:
    feats = {
        "INDE": 7.5,
        "IDA": 0.0,
        "IEG": 0.0,
        "IPV": 5.0,
    }
    assert ficha_suspect_missing_indicators(feats) is True


def test_not_suspect_when_only_one_axis_zero() -> None:
    feats = {
        "INDE": 7.5,
        "IDA": 0.0,
        "IEG": 8.0,
        "IPV": 6.0,
    }
    assert ficha_suspect_missing_indicators(feats) is False


def test_suspect_when_two_of_three_zero_not_ida_ieg() -> None:
    feats = {
        "INDE": 5.0,
        "IDA": 0.0,
        "IEG": 3.0,
        "IPV": 0.0,
    }
    assert ficha_suspect_missing_indicators(feats) is True


def test_not_suspect_when_inde_not_plausible() -> None:
    assert ficha_suspect_missing_indicators({"INDE": 0.0, "IDA": 0.0, "IEG": 0.0}) is False
    assert ficha_suspect_missing_indicators({"INDE": 10.0, "IDA": 0.0, "IEG": 0.0}) is False


def test_extra_lines_include_quality_heading() -> None:
    feats = {"INDE": 7.5, "IDA": 0.0, "IEG": 0.0, "IPV": 1.0}
    extra = ficha_quality_snapshot_extra_lines(feats)
    assert any("Qualidade dos dados na ficha" in line for line in extra)
    assert any("lacuna" in line for line in extra)


def test_is_zero_pedagogy_score() -> None:
    assert is_zero_pedagogy_score(0.0) is True
    assert is_zero_pedagogy_score(0.0001) is False
    assert is_zero_pedagogy_score(None) is False
