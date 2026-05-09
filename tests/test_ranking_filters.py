from __future__ import annotations

import pandas as pd

from passos_magico.ml.features import (
    latest_row_per_ra_table,
    one_row_per_ra_for_year,
    reference_years_available,
    rows_for_reference_year,
)
from passos_magico.ui.ranking_filters import (
    fase_as_int,
    ranking_fase_options,
    ranking_mask,
    ranking_turma_letter_options,
    ranking_turma_letter_options_for_fase,
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


def test_ranking_mask_fase_none_all_phases() -> None:
    df = pd.DataFrame(
        {
            "Fase": [6, 8, 8],
            "Turma": ["A", "B", "B"],
            "RA": ["x", "y", "z"],
        }
    )
    assert ranking_mask(df, None, "").tolist() == [True, True, True]
    m = ranking_mask(df, None, "B")
    assert m.tolist() == [False, True, True]


def test_ranking_turma_letter_options_for_fase_subset() -> None:
    df = pd.DataFrame(
        {
            "Fase": [1, 1, 2, 2],
            "Turma": ["A", "B", "B", "C"],
        }
    )
    assert ranking_turma_letter_options_for_fase(df, None) == ["A", "B", "C"]
    assert ranking_turma_letter_options_for_fase(df, 1) == ["A", "B"]
    assert ranking_turma_letter_options_for_fase(df, 2) == ["B", "C"]


def test_global_latest_then_fase_filter_matches_individual_ficha() -> None:
    """Matriz «um por aluno» deve usar último ano em **todo** o ficheiro, não o último só dentro do filtro."""
    df = pd.DataFrame(
        {
            "RA": ["X", "X"],
            "Nome": ["Aluno", "Aluno"],
            "Fase": [3, 6],
            "Turma": ["A", "A"],
            "Ano": [2020, 2023],
        }
    )
    latest = latest_row_per_ra_table(df)
    assert len(latest) == 1
    assert int(latest.iloc[0]["Ano"]) == 2023
    assert int(latest.iloc[0]["Fase"]) == 6
    # Ficha individual olharia Fase 6 — filtro «Fase 3» não deve incluir este RA na base latest
    assert not bool(ranking_mask(latest, 3, "").iloc[0])
    assert bool(ranking_mask(latest, 6, "").iloc[0])


def test_reference_year_slice_and_one_per_ra() -> None:
    df = pd.DataFrame(
        {
            "RA": ["a", "a", "b"],
            "Nome": ["A", "A", "B"],
            "Fase": [1, 2, 1],
            "Turma": ["A", "A", "B"],
            "Ano": [2023, 2024, 2024],
        }
    )
    assert reference_years_available(df) == [2024, 2023]
    r24 = rows_for_reference_year(df, 2024)
    assert len(r24) == 2
    one24 = one_row_per_ra_for_year(df, 2024)
    assert len(one24) == 2
    assert set(one24["RA"].astype(str).tolist()) == {"a", "b"}
