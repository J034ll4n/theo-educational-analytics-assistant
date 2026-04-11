"""Engenharia de features alinhada ao treino e ao simulador What-If."""

from __future__ import annotations

import numpy as np
import pandas as pd

PEDRA_MAP = {
    "Quartzo": 1,
    "Ágata": 2,
    "Agata": 2,
    "Ametista": 3,
    "Topázio": 4,
    "Topazio": 4,
}

TURMA_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}


def encode_pedra(val) -> float:
    if pd.isna(val):
        return 2.0
    s = str(val).strip()
    return float(PEDRA_MAP.get(s, 2))


def encode_turma(val) -> float:
    if pd.isna(val):
        return 2.0
    s = str(val).strip().upper()[:1]
    return float(TURMA_MAP.get(s, 2))


FEATURE_ORDER: list[str] = [
    "Fase",
    "Turma_ord",
    "Ano",
    "INDE",
    "IDA",
    "IAN",
    "IEG",
    "IPV",
    "Pedra_ord",
]


def augment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Pedra" in out.columns:
        out["Pedra_ord"] = out["Pedra"].apply(encode_pedra)
    else:
        out["Pedra_ord"] = 2.0
    if "Turma" in out.columns:
        out["Turma_ord"] = out["Turma"].apply(encode_turma)
    else:
        out["Turma_ord"] = 2.0
    return out


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Define rótulo alto_risco por regra pedagógica simples (snapshot estático)."""
    d = augment_dataframe(df)
    # Risco: aprendizado baixo e engajamento baixo, ou INDE muito baixo
    cond = (
        (d["IDA"] < 6.5) & (d["IEG"] < 6.0)
    ) | (d["INDE"] < 5.0)
    y = cond.astype(int).to_numpy()
    X = d[FEATURE_ORDER].astype(float)
    return X, y


def vector_from_values(
    fase: float,
    turma_ord: float,
    ano: float,
    inde: float,
    ida: float,
    ian: float,
    ieg: float,
    ipv: float,
    pedra_ord: float,
) -> np.ndarray:
    return np.array(
        [[fase, turma_ord, ano, inde, ida, ian, ieg, ipv, pedra_ord]],
        dtype=float,
    )


def row_features_from_df(df: pd.DataFrame, ra: str) -> dict[str, float] | None:
    sub = df[df["RA"].astype(str) == str(ra)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    aug = augment_dataframe(sub.head(1))
    r = aug.iloc[0]
    return {
        "Fase": float(r["Fase"]),
        "Turma_ord": float(r["Turma_ord"]),
        "Ano": float(r["Ano"]),
        "INDE": float(r["INDE"]),
        "IDA": float(r["IDA"]),
        "IAN": float(r["IAN"]),
        "IEG": float(r["IEG"]),
        "IPV": float(r["IPV"]),
        "Pedra_ord": float(r["Pedra_ord"]),
        "Nome": str(row.get("Nome", "")),
    }
