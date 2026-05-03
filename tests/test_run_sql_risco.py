"""Coluna `risco` injetada para DuckDB e normalização de operadores SQL."""

from __future__ import annotations

import numpy as np
import pandas as pd

from passos_magico.data_engine.query import normalize_sql_comparison_operators, run_sql, validate_select_only
from passos_magico.ml.inference import ensure_risco_column


def test_normalize_unicode_operators() -> None:
    s = "SELECT COUNT(*) FROM dados WHERE risco ≥ 0.46"
    assert ">=" in normalize_sql_comparison_operators(s)
    assert "≥" not in normalize_sql_comparison_operators(s)


def test_ensure_risco_column_adds_from_probs(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "RA": ["RA1", "RA2"],
            "Nome": ["A", "B"],
            "Fase": [6, 6],
            "Turma": ["A", "B"],
            "Ano": [2022, 2022],
            "INDE": [7.0, 5.0],
            "IDA": [6.0, 4.0],
            "IAN": [7.0, 6.0],
            "IEG": [7.0, 5.0],
            "IPV": [6.0, 6.0],
            "Pedra": ["Quartzo", "Ágata"],
        }
    )

    def _fake(_bundle, d: pd.DataFrame) -> np.ndarray:
        assert len(d) == 2
        return np.array([0.3, 0.72], dtype=np.float64)

    monkeypatch.setattr(
        "passos_magico.ml.inference.predict_risk_probabilities",
        _fake,
    )
    out = ensure_risco_column(df, object())
    assert "risco" in out.columns
    assert list(out["risco"].round(4)) == [0.3, 0.72]
    assert "risco" not in df.columns


def test_run_sql_normalizes_unicode_operator(monkeypatch) -> None:
    df = pd.DataFrame({"RA": ["x"], "x": [1]})
    monkeypatch.setattr(
        "passos_magico.ml.inference.ensure_risco_column",
        lambda d, _b: d,
    )
    sql = 'SELECT COUNT(*) AS n FROM dados WHERE "x" ≥ 1'
    out = run_sql(sql, df=df, bundle=None)
    assert int(out.iloc[0]["n"]) == 1


def test_run_sql_count_high_risk_with_injected_risco(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "RA": ["RA1", "RA2", "RA3"],
            "Nome": ["A", "B", "C"],
            "Fase": [8, 8, 8],
            "Turma": ["A", "B", "C"],
            "Ano": [2022, 2022, 2022],
            "INDE": [7.0, 6.0, 5.0],
            "IDA": [6.0, 5.0, 4.0],
            "IAN": [7.0, 6.0, 5.0],
            "IEG": [7.0, 6.0, 5.0],
            "IPV": [6.0, 6.0, 6.0],
            "Pedra": ["Quartzo", "Quartzo", "Ágata"],
        }
    )

    monkeypatch.setattr(
        "passos_magico.ml.inference.predict_risk_probabilities",
        lambda _b, d: np.array([0.5, 0.4, 0.6], dtype=np.float64),
    )
    df_enriched = ensure_risco_column(df.copy(), object())
    sql = "SELECT COUNT(*) AS n FROM dados WHERE risco >= 0.46"
    ok, err = validate_select_only(sql)
    assert ok, err
    result = run_sql(sql, df=df_enriched, bundle=None)
    assert int(result.iloc[0]["n"]) == 2
