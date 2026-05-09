"""Bloco de KPIs derivados do DataFrame para o prompt de insight (sem LLM)."""

from __future__ import annotations

import re

import pandas as pd

_COUNT_HINTS = ("total", "quantidade", "n_registos", "n_alunos", "count", "alunos", "freq", "n_")
_INDICATOR_SUBSTR = ("inde", "ida", "ieg", "ipv", "ian")

# Colunas úteis para narrativa de ficha (evita SELECT * com dezenas de métricas técnicas).
_STUDENT_SERIES_COLS: tuple[str, ...] = (
    "Ano",
    "Fase",
    "Turma",
    "Pedra",
    "INDE",
    "IDA",
    "IAN",
    "IEG",
    "IPV",
    "IAA",
    "IPS",
    "IPP",
    "MAT",
    "POR",
    "risco",
    "Nome",
    "data_nasc",
    "Genero",
)

_EXCLUDE_TECH_NAMES = frozenset(
    {
        "cg",
        "cf",
        "ct",
        "ano_ingresso",
        "instituicao_de_ensino",
        "escola",
        "serie_escolar",
    }
)


def _column_looks_scale_indicator(name: str) -> bool:
    nl = str(name).lower()
    return any(x in nl for x in _INDICATOR_SUBSTR)


def _degenerate_indicator_alerts(df: pd.DataFrame, numeric_pairs: list[tuple[str, pd.Series]]) -> list[str]:
    """Avisos quando indicadores 0–10 vêm todos ~0 ou só nulas (evita narrar «pior turma» falso)."""
    out: list[str] = []
    if len(df) < 2:
        return out
    for c, s in numeric_pairs:
        if not _column_looks_scale_indicator(c):
            continue
        sn = s.dropna()
        if sn.empty:
            out.append(
                f"**Alerta:** a coluna **{c}** está toda nula nesta amostra — o ranking **não** reflete desempenho medido."
            )
            break
        mx, mn = float(sn.max()), float(sn.min())
        if mx < 0.05 and mn >= -1e-9:
            out.append(
                f"**Alerta:** **{c}** vem ≤ 0,05 em todas as linhas — provável **lacuna de dados** ou filtro; "
                "**não** conclua «INDE zero real» em todas as turmas sem validar o Parquet (p.ex. `Ano`, `INDE` preenchido)."
            )
            break
    return out


def _is_count_like_column(name: str) -> bool:
    nl = str(name).lower()
    return any(k in nl for k in _COUNT_HINTS)


def _fmt_scalar_num(v: float) -> str:
    if pd.notna(v) and abs(v - round(v)) < 1e-9 and abs(v) < 1e12:
        return str(int(round(v)))
    return f"{v:.4g}"


def _fmt_cell(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        fv = float(v)
        if pd.notna(fv) and abs(fv - round(fv)) < 1e-9 and abs(fv) < 1e12:
            return str(int(round(fv)))
        if pd.notna(fv):
            return f"{fv:.3g}"
    return str(v)


def _exclude_from_student_aux_column(name: str) -> bool:
    nl = str(name).strip().lower()
    if nl in _EXCLUDE_TECH_NAMES:
        return True
    if re.match(r"^pedra_20\d{2}$", nl):
        return True
    if re.match(r"^inde_20\d{2}$", nl):
        return True
    if re.match(r"^pedra_\d{4}$", nl):
        return True
    return False


def _looks_boolean_zero_one(s: pd.Series) -> bool:
    sn = pd.to_numeric(s, errors="coerce").dropna()
    if len(sn) < 1:
        return False
    try:
        vals = {round(float(x), 6) for x in sn.unique()}
    except (TypeError, ValueError):
        return False
    return vals <= {0.0, 1.0}


def _is_single_student_sample(df: pd.DataFrame) -> bool:
    if df is None or df.empty or "RA" not in df.columns:
        return False
    return int(df["RA"].nunique(dropna=True)) == 1


def _nome_e_placeholder_generico(nome: str, ra: str) -> bool:
    n = str(nome).strip()
    if not n:
        return True
    if re.match(r"^aluno-\d+$", n, re.IGNORECASE):
        return True
    suffix = ra.replace("RA-", "").replace("RA", "").strip("-")
    if suffix and n.lower() == f"aluno-{suffix}".lower():
        return True
    return False


def _kpi_narration_single_student(df: pd.DataFrame, *, max_rows: int) -> str | None:
    """Resumo para uma única coluna RA — sem «médias» sobre cg/cf ou booleanos."""
    ra = str(df["RA"].dropna().iloc[0])
    n = len(df)
    lines: list[str] = [
        f"**Perfil (um aluno):** **{ra}** — **{n}** linha(s) devolvida(s) pela consulta."
    ]

    if "Nome" in df.columns:
        raw = str(df["Nome"].dropna().iloc[0]) if df["Nome"].notna().any() else ""
        if raw and not _nome_e_placeholder_generico(raw, ra):
            lines.append(f"Nome na base: **{raw}**.")
        elif raw:
            lines.append(
                "O **Nome** na base é genérico (ex.: «Aluno-…») — **não** trate como nome civil; "
                "na resposta use sobretudo o **RA**."
            )

    dord = df.sort_values("Ano") if "Ano" in df.columns else df.copy()

    if "Ano" in dord.columns and len(dord) > 1:
        evo: list[str] = []
        for _, row in dord.head(max_rows).iterrows():
            ano_v = row.get("Ano")
            if pd.isna(ano_v):
                continue
            try:
                ano_i = int(round(float(ano_v)))
            except (TypeError, ValueError):
                continue
            bits: list[str] = [f"**Ano {ano_i}**"]
            for c in ("Fase", "Turma", "Pedra", "INDE", "IDA", "IAN", "IEG", "IPV", "risco"):
                if c not in dord.columns:
                    continue
                v = row.get(c)
                if pd.notna(v):
                    bits.append(f"{c}={_fmt_cell(v)}")
            evo.append(": ".join(bits[:1]) + (" — " + ", ".join(bits[1:]) if len(bits) > 1 else ""))
        if evo:
            lines.append("Evolução (por ano letivo):")
            lines.extend(f"- {e}" for e in evo)
    elif len(dord) == 1:
        row = dord.iloc[0]
        bits = []
        for c in _STUDENT_SERIES_COLS:
            if c not in dord.columns or c == "Nome":
                continue
            v = row.get(c)
            if pd.notna(v):
                bits.append(f"**{c}:** {_fmt_cell(v)}")
        if bits:
            lines.append("Último registo: " + "; ".join(bits[:14]))

    for c in sorted(df.columns, key=str.lower):
        if c in ("RA",) or c in _STUDENT_SERIES_COLS:
            continue
        if _exclude_from_student_aux_column(c):
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().sum() == 0:
            continue
        if not _looks_boolean_zero_one(sn):
            continue
        last = float(sn.iloc[-1])
        const = sn.nunique(dropna=True) <= 1
        lab = "**Sim**" if last >= 0.5 else "**Não**"
        if const:
            lines.append(f"**{c}:** sempre {lab} (campo 0/1).")
        else:
            lines.append(f"**{c}:** último ano na amostra → {lab} (campo 0/1; **não** chame isto de «média»).")

    lines.append(
        "**Instrução ao modelo:** **não** descreva como «média de desempenho» colunas técnicas "
        "(`cg`, `cf`, `ct`, `ano_ingresso`, `pedra_20xx`, `inde_20xx`, etc.) nem use a palavra **média** "
        "para campos **booleanos** (0/1) — use **Sim/Não** ou «constante»."
    )
    return "\n".join(lines)


def kpi_narration_block(df: pd.DataFrame, *, max_rows: int = 15) -> str | None:
    """Resumo numérico derivado só do DataFrame (rastreável). Retorna None se não aplicável."""
    if df is None or df.empty or len(df) > max_rows:
        return None

    if _is_single_student_sample(df):
        return _kpi_narration_single_student(df, max_rows=max_rows)

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
                lines.append(f"**{c}**: **{_fmt_scalar_num(v)}**.")

    if len(df) == 1:
        for c, s in numeric_pairs[:12]:
            v = float(s.iloc[0])
            if pd.notna(v) and c not in counted:
                lines.append(f"**{c}**: {_fmt_scalar_num(v)}.")
        text = "\n".join(lines)
        return text if len(lines) > 1 else None

    for c, s in numeric_pairs[:10]:
        if c in counted:
            continue
        lines.append(
            f"**{c}**: média {float(s.mean()):.2f} "
            f"(mín. {float(s.min()):.2f}, máx. {float(s.max()):.2f})."
        )

    lines.extend(_degenerate_indicator_alerts(df, numeric_pairs)[:2])

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

    text = "\n".join(lines[:20])
    return text if len(lines) > 1 else None
