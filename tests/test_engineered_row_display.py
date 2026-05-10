"""Linha engenharada para trajetória e simulador: ano coerente com `Ano` vs `ano_referencia`."""

from __future__ import annotations

import pandas as pd

from passos_magico.ml.risk_pipeline import engineered_row_for_display, ensure_risk_engineering


def test_engineered_row_prefers_an_col_when_conflict_with_ano_referencia() -> None:
    """Se `Ano` e `ano_referencia` divergem, seguir a mesma prioridade que o restante da base (`Ano` primeiro)."""
    rows = [
        {
            "RA": "X1",
            "Nome": "A",
            "Fase": 3,
            "Turma": "A",
            "Pedra": "Ágata",
            "Instituicao_de_ensino": "Escola 1",
            "Genero": "M",
            "Ano": 2024,
            "ano_referencia": 2023,
            "inde": 6.6,
            "ida": 5.5,
            "iaa": 10.0,
            "iec": 5.0,
        },
        {
            "RA": "X1",
            "Nome": "A",
            "Fase": 3,
            "Turma": "A",
            "Pedra": "Ágata",
            "Instituicao_de_ensino": "Escola 1",
            "Genero": "M",
            "Ano": 2023,
            "ano_referencia": 2023,
            "inde": 3.0,
            "ida": 4.0,
            "iaa": 4.0,
            "iec": 5.0,
        },
    ]
    df = pd.DataFrame(rows)
    df = ensure_risk_engineering(df)

    hit = engineered_row_for_display(df, "X1", 2024)
    assert hit is not None
    assert abs(float(hit["inde"]) - 6.6) < 1e-6
    assert abs(float(hit["iaa"]) - 10.0) < 1e-6


def test_engineered_row_falls_back_to_ano_referencia_if_an_missing() -> None:
    df = pd.DataFrame(
        [
            {
                "RA": "Y1",
                "Nome": "B",
                "Fase": 1,
                "Turma": "B",
                "Pedra": "Ágata",
                "Instituicao_de_ensino": "Escola 1",
                "Genero": "F",
                "ano_referencia": 2024,
                "inde": 7.2,
                "ida": 6.1,
                "iaa": 6.5,
                "iec": 5.0,
            },
        ]
    )
    df = ensure_risk_engineering(df)

    hit = engineered_row_for_display(df, "Y1", 2024)
    assert hit is not None
    assert abs(float(hit["inde"]) - 7.2) < 1e-6
