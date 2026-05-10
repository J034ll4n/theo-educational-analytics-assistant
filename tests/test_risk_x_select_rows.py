"""`risk_X_matrix_select_rows` alinha subconjuntos ao `risk_X_matrix` da base completa (RA + ano)."""

from unittest.mock import patch

import numpy as np
import pandas as pd


def test_risk_X_matrix_select_rows_picks_row_by_ra_and_year():
    full_df = pd.DataFrame(
        {
            "RA": ["1", "2", "1"],
            "ano_referencia": [2023, 2023, 2024],
        }
    )
    full_df.index = [10, 20, 30]
    sub = pd.DataFrame({"RA": ["1"], "ano_referencia": [2024]})

    def fake_risk_X_matrix(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"feat": np.arange(len(df), dtype=float)}, index=df.index)

    with patch("passos_magico.ml.risk_pipeline.risk_X_matrix", side_effect=fake_risk_X_matrix):
        from passos_magico.ml.risk_pipeline import risk_X_matrix_select_rows

        out = risk_X_matrix_select_rows(full_df, sub)
    assert len(out) == 1
    assert list(out.index) == [30]
    assert float(out.iloc[0, 0]) == 2.0


def test_risk_X_matrix_select_rows_preserves_sub_order():
    full_df = pd.DataFrame(
        {
            "RA": ["a", "b"],
            "Ano": [2024, 2024],
        }
    )
    full_df.index = [100, 200]
    sub = pd.DataFrame(
        {
            "RA": ["b", "a"],
            "Ano": [2024, 2024],
        }
    )

    def fake_risk_X_matrix(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"feat": df.index.astype(float)}, index=df.index)

    with patch("passos_magico.ml.risk_pipeline.risk_X_matrix", side_effect=fake_risk_X_matrix):
        from passos_magico.ml.risk_pipeline import risk_X_matrix_select_rows

        out = risk_X_matrix_select_rows(full_df, sub)
    assert list(out.index) == [200, 100]
