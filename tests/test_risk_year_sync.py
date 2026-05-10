"""Alinhamento ano ficha individual vs filtro da matriz de priorização."""

from passos_magico.ml.features import default_ficha_year_synced_with_matrix


def test_default_ficha_year_when_matrix_all_years_uses_latest_ra_year():
    assert default_ficha_year_synced_with_matrix([2025, 2024, 2023], None) == 2025


def test_default_ficha_year_when_matrix_fixed_year_matches():
    assert default_ficha_year_synced_with_matrix([2025, 2024, 2023], 2024) == 2024


def test_default_ficha_year_matrix_year_not_in_ra_falls_back_to_latest():
    assert default_ficha_year_synced_with_matrix([2023, 2022], 2024) == 2023
