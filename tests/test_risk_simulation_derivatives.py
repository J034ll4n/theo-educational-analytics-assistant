"""Derivadas do simulador alinhadas ao notebook (choque, distância turma, Δ INDE)."""

from __future__ import annotations

import pandas as pd

from passos_magico.ml.features import single_row_for_ra_and_year

from passos_magico.ml.risk_pipeline import (
    RISK_MODEL_COLUMNS,
    build_X_after_slider_simulation,
    ensure_risk_engineering,
)


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


def test_simulation_row_matches_sim_ano_not_only_latest_ra() -> None:
    """`build_X_after_slider_simulation` deve usar RA+ano de `sim`, não sempre o último ano na base."""
    older = _df_one_student().iloc[[0]].copy()
    older["inde"] = 4.0
    older["idade"] = 13.25
    older["ano_referencia"] = 2023
    older["Ano"] = 2023

    newer = _df_one_student().iloc[[0]].copy()
    newer["inde"] = 9.5
    newer["idade"] = 99.875
    newer["ano_referencia"] = 2024
    newer["Ano"] = 2024

    df_raw = pd.concat([older, newer], axis=0, ignore_index=True)
    df_eng = ensure_risk_engineering(df_raw.copy())

    yo = single_row_for_ra_and_year(df_eng, "TEST_RA_SIM", 2023).iloc[0]
    yn = single_row_for_ra_and_year(df_eng, "TEST_RA_SIM", 2024).iloc[0]
    dy = abs(float(yn["idade"]) - float(yo["idade"]))
    assert dy > 1e-9, "teste mal moldado — precisamos de linhas distintas entre anos"

    sim_old = {
        "Fase": 5.0,
        "Turma_ord": 2.0,
        "Ano": 2023.0,
        "INDE": float(yn["inde"]),
        "IDA": float(yn["ida"]),
        "IAN": float(yn["ian"]),
        "IEG": float(yn["ieg"]),
        "IPV": float(yn["ipv"]),
        "Pedra_ord": 2.0,
        "IAA": float(yn["iaa"]),
        "IPS": float(yn["ips"]),
        "IPP": float(yn["ipp"]),
        "ING": float(yn["ing"]),
        "MAT": float(yn["mat"]),
        "POR": float(yn["por"]),
        "Delta_INDE": float(yo["delta_inde"]),
    }
    xm = build_X_after_slider_simulation(df_raw, "TEST_RA_SIM", sim_old)
    assert xm is not None and len(xm) == 1
    assert abs(float(xm.iloc[0]["idade"]) - float(yo["idade"])) < 1e-5


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
        "IPP": 6.0,
        "ING": 6.0,
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


def test_build_x_applies_ipv_ipp_ing_from_sim() -> None:
    """IPV, IPP e ING do dicionário `sim` devem substituir a linha antes da matriz do modelo."""
    df = _df_one_student()
    sim = {
        "Fase": 5.0,
        "Turma_ord": 2.0,
        "Ano": 2024.0,
        "INDE": 6.0,
        "IDA": 6.0,
        "IAN": 6.0,
        "IEG": 7.0,
        "IPV": 8.5,
        "Pedra_ord": 2.0,
        "IAA": 6.0,
        "IPS": 6.0,
        "IPP": 3.25,
        "ING": 9.0,
        "MAT": 6.0,
        "POR": 6.0,
        "Delta_INDE": 0.0,
    }
    xm = build_X_after_slider_simulation(df, "TEST_RA_SIM", sim)
    assert xm is not None and len(xm) == 1
    r = xm.iloc[0]
    assert abs(float(r["ipv"]) - 8.5) < 1e-5
    assert abs(float(r["ipp"]) - 3.25) < 1e-5
    assert abs(float(r["ing"]) - 9.0) < 1e-5


def test_ensure_risk_engineering_fills_ipp_from_cf_ct() -> None:
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
        elif c == "ipp":
            continue
        else:
            row[c] = 5.0
    row["inde"] = 6.0
    row["ra"] = "R1"
    row["ano_referencia"] = 2024
    row["RA"] = "R1"
    row["Ano"] = 2024
    row["cf"] = 6.0
    row["ct"] = 8.0
    d = pd.DataFrame([row])
    eng = ensure_risk_engineering(d)
    assert abs(float(eng.iloc[0]["ipp"]) - 7.0) < 1e-5
