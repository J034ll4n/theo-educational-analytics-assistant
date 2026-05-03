"""Bloco de KPIs derivados do DataFrame para o prompt de insight (sem LLM)."""

from __future__ import annotations

import pandas as pd

_COUNT_HINTS = ("total", "quantidade", "n_alunos", "count", "alunos", "freq", "n_")


def _is_count_like_column(name: str) -> bool:
    nl = str(name).lower()
    return any(k in nl for k in _COUNT_HINTS)


def kpi_narration_block(df: pd.DataFrame, *, max_rows: int = 15) -> str | None:
    """Resumo numérico derivado só do DataFrame (rastreável). Retorna None se não aplicável."""
    if df is None or df.empty or len(df) > max_rows:
        return None

    numeric_pairs: list[tuple[str, pd.Series]] = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_pairs.append((c, s))

    if not numeric_pairs:
        return None

    lines: list[str] = [f"A consulta devolveu **{len(df)}** linha(s)."]

    counted: set[str] = set()
    for c, s in numeric_pairs:
        if not _is_count_like_column(c):
            continue
        counted.add(c)
        if len(df) > 1:
            lines.append(f"Soma de **{c}** na amostra: **{float(s.sum()):.0f}**.")
        else:
            v = float(s.iloc[0])
            if pd.notna(v):
                lines.append(f"**{c}**: **{v:.2f}**.")

    if len(df) == 1:
        for c, s in numeric_pairs[:12]:
            v = float(s.iloc[0])
            if pd.notna(v) and c not in counted:
                lines.append(f"**{c}**: {v:.2f}.")
        text = "\n".join(lines)
        return text if len(lines) > 1 else None

    for c, s in numeric_pairs[:10]:
        if c in counted:
            continue
        lines.append(
            f"**{c}**: média {float(s.mean()):.2f} "
            f"(mín. {float(s.min()):.2f}, máx. {float(s.max()):.2f})."
        )

    primary: tuple[str, pd.Series] | None = None
    for c, s in numeric_pairs:
        if c not in counted:
            primary = (c, s)
            break
    if primary is None:
        primary = numeric_pairs[0]
    c0, s0 = primary
    try:
        idx = s0.idxmax()
        mx = float(s0.loc[idx])
        row = df.loc[idx]
        dims = [
            str(row[d])
            for d in df.columns
            if d != c0 and d in row.index and pd.notna(row[d])
        ][:4]
        lbl = ", ".join(dims) if dims else "linha correspondente"
        lines.append(f"Pico de **{c0}** nesta amostra: **{mx:.2f}** ({lbl}).")
    except (TypeError, KeyError, ValueError):
        pass

    text = "\n".join(lines[:14])
    return text if len(lines) > 1 else None
