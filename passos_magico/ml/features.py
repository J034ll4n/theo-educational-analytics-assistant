"""Engenharia de features alinhada ao treino e ao simulador What-If."""

from __future__ import annotations

from typing import Any

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


def _year_series_for_sort(df: pd.DataFrame) -> pd.Series:
    """Ano numérico para ordenar (mais recente = maior)."""
    for c in ("Ano", "ano_referencia"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def pick_latest_year_row(sub: pd.DataFrame) -> pd.DataFrame:
    """Uma linha do subconjunto: maior ano de referência; empate → primeira linha."""
    if sub.empty:
        return sub
    if len(sub) == 1:
        return sub.iloc[[0]]
    y = _year_series_for_sort(sub)
    if y.notna().any():
        max_y = y.max()
        tie = sub.loc[y == max_y]
        return tie.iloc[[0]]
    return sub.iloc[[0]]


def latest_single_row_for_ra(df: pd.DataFrame, ra: str) -> pd.DataFrame:
    """Filtra por RA e devolve uma linha (último ano disponível na base)."""
    ra_col = "RA" if "RA" in df.columns else "ra"
    if ra_col not in df.columns:
        return pd.DataFrame()
    sub = df[df[ra_col].astype(str) == str(ra)]
    return pick_latest_year_row(sub)


def latest_row_per_ra_table(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por RA (último ano); para listas na UI sem duplicar aluno."""
    ra_col = "RA" if "RA" in df.columns else "ra"
    if ra_col not in df.columns or df.empty:
        return df
    work = df.copy()
    work["_y"] = _year_series_for_sort(work)
    work = work.sort_values("_y", ascending=False, na_position="last")
    out = work.drop_duplicates(subset=[ra_col], keep="first")
    return out.drop(columns=["_y"], errors="ignore")


def reference_years_available(df: pd.DataFrame) -> list[int]:
    """Anos de referência distintos na base (mais recente primeiro)."""
    if df.empty:
        return []
    y = _year_series_for_sort(df).dropna()
    if y.empty:
        return []
    return sorted({int(round(float(v))) for v in y.unique()}, reverse=True)


def years_for_ra(df: pd.DataFrame, ra: str) -> list[int]:
    """Anos disponíveis para este RA (mais recente primeiro)."""
    ra_col = "RA" if "RA" in df.columns else "ra"
    if ra_col not in df.columns or not str(ra).strip():
        return []
    sub = df[df[ra_col].astype(str) == str(ra)]
    if sub.empty:
        return []
    y = _year_series_for_sort(sub).dropna()
    if y.empty:
        return []
    return sorted({int(round(float(v))) for v in y.unique()}, reverse=True)


def previous_reference_year(df: pd.DataFrame, ra: str, current_year: int) -> int | None:
    """Maior ano de referência estritamente anterior a `current_year` para este RA."""
    ys = sorted(set(years_for_ra(df, ra)))
    older = [u for u in ys if u < int(current_year)]
    return max(older) if older else None


def rows_for_reference_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Todas as linhas com ano de referência igual a `year`."""
    if df.empty:
        return df.iloc[0:0].copy()
    y = _year_series_for_sort(df)
    return df.loc[y == int(year)].copy()


def one_row_per_ra_for_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """No máximo uma linha por RA, todas com o mesmo ano de referência."""
    sub = rows_for_reference_year(df, year)
    if sub.empty:
        return sub
    ra_col = "RA" if "RA" in sub.columns else "ra"
    if ra_col not in sub.columns:
        return sub.iloc[0:0].copy()
    return sub.drop_duplicates(subset=[ra_col], keep="first")


def single_row_for_ra_and_year(df: pd.DataFrame, ra: str, ref_year: int) -> pd.DataFrame:
    """Uma linha: RA + ano de referência exatos (empate → primeira)."""
    ra_col = "RA" if "RA" in df.columns else "ra"
    if ra_col not in df.columns:
        return pd.DataFrame()
    sub = df[df[ra_col].astype(str) == str(ra)]
    if sub.empty:
        return pd.DataFrame()
    y = _year_series_for_sort(sub)
    match = sub.loc[y == int(ref_year)]
    if match.empty:
        return pd.DataFrame()
    return match.iloc[[0]]


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


def _row_ano_float(row: pd.Series) -> float:
    for c in ("Ano", "ano_referencia"):
        if c in row.index and pd.notna(row.get(c)):
            try:
                return float(row[c])
            except (TypeError, ValueError):
                continue
    return float("nan")


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


def row_features_from_df(df: pd.DataFrame, ra: str, ref_year: int | None = None) -> dict[str, Any] | None:
    """Features de uma linha do aluno. `ref_year=None` → último ano disponível na base."""
    ra_col = "RA" if "RA" in df.columns else "ra"
    if ra_col not in df.columns:
        return None
    if ref_year is None:
        sub = latest_single_row_for_ra(df, ra)
    else:
        sub = single_row_for_ra_and_year(df, ra, int(ref_year))
    if sub.empty:
        return None
    row = sub.iloc[0]
    aug = augment_dataframe(sub)
    r = aug.iloc[0]
    ano_val = _row_ano_float(row)
    return {
        "RA": str(ra),
        "Fase": float(r["Fase"]),
        "Turma_ord": float(r["Turma_ord"]),
        "Ano": ano_val,
        "INDE": float(r["INDE"]),
        "IDA": float(r["IDA"]),
        "IAN": float(r["IAN"]),
        "IEG": float(r["IEG"]),
        "IPV": float(r["IPV"]),
        "Pedra_ord": float(r["Pedra_ord"]),
        "Nome": str(row.get("Nome", "")),
    }
