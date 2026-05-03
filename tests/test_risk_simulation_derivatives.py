"""Derivadas do simulador alinhadas ao notebook (choque, distância turma, Δ INDE)."""

from __future__ import annotations

import pandas as pd

from passos_magico.ml.risk_pipeline import RISK_MODEL_COLUMNS, build_X_after_slider_simulation


def _df_one_student() -> pd.DataFrame:
    row: dict = {}
    for c in RISK_MODEL_COLUMNS:
        if c == "fase":
            row[c] = "Fase 5"
        elif c == "genero":
            row[c] = "M"
        elif c == "instituicao_de_ensino":
            row[c] = "Escola X"
        elif c == "pedra":
            row[c] = "Ágata"
        else:
            row[c] = 5.0
    row["inde"] = 6.0
    row["ida"] = 6.0
    row["iaa"] = 8.0
    row["ieg"] = 7.0
    row["ipv"] = 6.0
    row["media_turma_inde"] = 6.5
    row["delta_inde"] = 0.1
    row["choque_realidade"] = 999.0
    row["distancia_media_turma"] = 999.0
    row["esforco_sem_resultado"] = 999.0
    df = pd.DataFrame([row])
    df["ra"] = "TEST_RA_SIM"
    df["ano_referencia"] = 2024
    df["RA"] = "TEST_RA_SIM"
    df["Ano"] = 2024
    df["Nome"] = "Aluno Teste"
    df["Fase"] = 5
    df["Turma"] = "B"
    return df


def test_build_x_recomputes_choque_distancia_delta_esforco() -> None:
    df = _df_one_student()
    sim = {
        "Fase": 5.0,
        "Turma_ord": 2.0,
        "Ano": 2024.0,
        "INDE": 7.0,
        "IDA": 4.0,
        "IAN": 6.0,
        "IEG": 7.0,
        "IPV": 6.0,
        "Pedra_ord": 2.0,
        "IAA": 9.0,
        "IPS": 6.0,
        "MAT": 6.0,
        "POR": 6.0,
        "Delta_INDE": 1.2,
    }
    xm = build_X_after_slider_simulation(df, "TEST_RA_SIM", sim)
    assert xm is not None and len(xm) == 1
    r = xm.iloc[0]
    assert abs(float(r["choque_realidade"]) - (9.0 - 4.0)) < 1e-5
    assert abs(float(r["distancia_media_turma"]) - (7.0 - 6.5)) < 1e-5
    assert abs(float(r["delta_inde"]) - 1.2) < 1e-5
    assert abs(float(r["esforco_sem_resultado"]) - (7.0 / (4.0 + 0.1))) < 1e-5
