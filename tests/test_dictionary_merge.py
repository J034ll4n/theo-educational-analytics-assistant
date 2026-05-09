from __future__ import annotations

import pandas as pd

from passos_magico.semantic.dictionary_merge import merge_dictionary_with_dataframe


def test_merge_preserves_user_description() -> None:
    df = pd.DataFrame({"RA": ["a"], "INDE": [7.0], "extra_col": [1]})
    rows = [
        {"coluna": "RA", "descricao": "MEU TEXTO PERSONALIZADO", "tipo": "string"},
        {"coluna": "INDE", "descricao": "", "tipo": "float64"},
    ]
    out = merge_dictionary_with_dataframe(df, rows)
    by = {r["coluna"]: r for r in out}
    assert by["RA"]["descricao"] == "MEU TEXTO PERSONALIZADO"
    assert "extra_col" in by
    assert len(out) == len(df.columns)


def test_merge_order_follows_dataframe() -> None:
    df = pd.DataFrame({"z": [1], "a": [2]})
    out = merge_dictionary_with_dataframe(df, [])
    assert [r["coluna"] for r in out] == ["z", "a"]


def test_merge_fills_stock_for_known_column() -> None:
    df = pd.DataFrame({"Ano": [2022], "risco": [0.5]})
    out = merge_dictionary_with_dataframe(df, [])
    by = {r["coluna"]: r for r in out}
    assert "ano letivo" in by["Ano"]["descricao"].lower() or "referência" in by["Ano"]["descricao"].lower()
    assert "0" in by["risco"]["descricao"] or "modelo" in by["risco"]["descricao"].lower()
