"""Filtros do ranking de risco: fase 1–8 e turma A–E, sem duplicados por grafia."""

from __future__ import annotations

import re

import pandas as pd

from passos_magico.ml.features import encode_turma

_TURMA_ORD_TO_LETTER: dict[int, str] = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
_LETTER_TO_ORD: dict[str, int] = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}


def fase_as_int(x) -> int:
    """Extrai fase 1–8; valores fora do intervalo ou inválidos → 0."""
    if pd.isna(x):
        return 0
    if isinstance(x, (int, float)) and not (isinstance(x, float) and pd.isna(x)):
        try:
            v = int(round(float(x)))
            return v if 1 <= v <= 8 else 0
        except (TypeError, ValueError):
            pass
    m = re.search(r"(\d+)", str(x).strip())
    if not m:
        return 0
    v = int(m.group(1))
    return v if 1 <= v <= 8 else 0


def series_fase_int(s: pd.Series) -> pd.Series:
    return s.map(fase_as_int)


def series_turma_ord(s: pd.Series) -> pd.Series:
    return s.map(encode_turma).round().astype(int).clip(1, 5)


def ranking_fase_options(df: pd.DataFrame) -> list[int]:
    """Fases distintas 1–8 presentes na base (ordenadas, sem duplicar por tipo)."""
    if "Fase" not in df.columns:
        return []
    u = sorted(set(series_fase_int(df["Fase"]).tolist()) - {0})
    return u


def ranking_turma_letter_options(df: pd.DataFrame) -> list[str]:
    """Letras A–E distintas presentes na base (normaliza « D », «d», etc.)."""
    if "Turma" not in df.columns:
        return []
    tcol = df["Turma"].dropna()
    if tcol.empty:
        return []
    so = tcol.map(encode_turma).round().astype(int).clip(1, 5)
    letters: set[str] = set()
    for o in so.dropna().unique():
        try:
            oi = int(o)
        except (TypeError, ValueError):
            continue
        lt = _TURMA_ORD_TO_LETTER.get(oi)
        if lt:
            letters.add(lt)
    return sorted(letters)


def ranking_turma_letter_options_for_fase(df: pd.DataFrame, fase: int | None) -> list[str]:
    """Letras de turma que **existem** no recorte (fase fixa ou todas as fases). Evita combinação fase×turma impossível na UI."""
    if fase is None:
        return ranking_turma_letter_options(df)
    if "Fase" not in df.columns or "Turma" not in df.columns:
        return ranking_turma_letter_options(df)
    sub = df.loc[series_fase_int(df["Fase"]) == int(fase)]
    if sub.empty:
        return []
    return ranking_turma_letter_options(sub)


def ranking_mask(df: pd.DataFrame, fase: int | None, turma_letter: str) -> pd.Series:
    """Máscara alinhada à normalização usada nas opções. `fase=None` = todas as fases."""
    if fase is not None and "Fase" not in df.columns:
        return pd.Series(False, index=df.index)
    if fase is None:
        fi = pd.Series(True, index=df.index)
    else:
        fi = series_fase_int(df["Fase"]) == int(fase)
    t = (turma_letter or "").strip().upper()[:1]
    if not t or t not in _LETTER_TO_ORD:
        return fi
    if "Turma" not in df.columns:
        return pd.Series(False, index=df.index)
    wo = _LETTER_TO_ORD[t]
    return fi & (series_turma_ord(df["Turma"]) == wo)
