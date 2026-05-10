"""Alinhamento ano ficha individual vs filtro da matriz de priorização."""

import pandas as pd

from passos_magico.ml.features import (
    default_ficha_year_synced_with_matrix,
    default_reference_year_operacional,
)


def test_default_ficha_year_when_matrix_all_years_uses_latest_ra_year():
    assert default_ficha_year_synced_with_matrix([2025, 2024, 2023], None) == 2025


def test_default_ficha_year_when_matrix_fixed_year_matches():
    assert default_ficha_year_synced_with_matrix([2025, 2024, 2023], 2024) == 2024


def test_default_ficha_year_matrix_year_not_in_ra_falls_back_to_latest():
    assert default_ficha_year_synced_with_matrix([2023, 2022], 2024) == 2023


def test_default_reference_year_operacional_prefers_2024_when_present():
    df = pd.DataFrame({"Ano": [2022, 2023, 2024], "RA": ["a", "b", "c"]})
    assert default_reference_year_operacional(df) == 2024


def test_default_reference_year_operacional_uses_latest_when_no_2024():
    df = pd.DataFrame({"Ano": [2022, 2023, 2025], "RA": ["a", "b", "c"]})
    assert default_reference_year_operacional(df) == 2025
