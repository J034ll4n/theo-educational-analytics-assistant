"""Catálogo smoke (20 perguntas) — sem Ollama."""

from __future__ import annotations

from tests.fixtures.theo_question_catalog import SMOKE_TWENTY_IDS, get_smoke_twenty_question_cases


def test_smoke_twenty_distinct_ids_and_length() -> None:
    cases = get_smoke_twenty_question_cases()
    assert len(cases) == len(SMOKE_TWENTY_IDS) == 20
    assert len({c["id"] for c in cases}) == 20
    assert all(c.get("pergunta") for c in cases)
