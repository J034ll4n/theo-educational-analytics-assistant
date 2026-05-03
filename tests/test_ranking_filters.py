from __future__ import annotations

import pandas as pd

from passos_magico.ui.ranking_filters import (
    fase_as_int,
    ranking_fase_options,
    ranking_mask,
    ranking_turma_letter_options,
)


def test_fase_as_int_unifies_string_and_number() -> None:
    assert fase_as_int(8) == 8
    assert fase_as_int(8.0) == 8
    assert fase_as_int("Fase 8") == 8
    assert fase_as_int("  6 ") == 6


def test_ranking_fase_options_dedupe_mixed_types() -> None:
    df = pd.DataFrame({"Fase": [8, "8", "Fase 6", 6.0, 8.0], "Turma": ["A"] * 5})
    assert ranking_fase_options(df) == [6, 8]


def test_ranking_turma_letters_normalize() -> None:
    df = pd.DataFrame({"Fase": [1, 1, 1, 1], "Turma": ["D", " d ", "D", "E"]})
    assert ranking_turma_letter_options(df) == ["D", "E"]


def test_ranking_mask_fase_string_turma_whitespace() -> None:
    df = pd.DataFrame(
        {
            "Fase": ["Fase 8", 8, 6],
            "Turma": [" D ", "C", "D"],
            "RA": ["a", "b", "c"],
        }
    )
    m = ranking_mask(df, 8, "D")
    assert m.tolist() == [True, False, False]
    m2 = ranking_mask(df, 8, "")
    assert m2.sum() == 2
