"""Motor DuckDB sobre Parquet — apenas SELECT."""

from __future__ import annotations

import re
import warnings
from collections import deque

import duckdb
import pandas as pd

from passos_magico.data_engine.loader import get_parquet_path, load_dados_df

_SELECT_RE = re.compile(
    r"^\s*select\b",
    re.IGNORECASE | re.DOTALL,
)
_WITH_RE = re.compile(
    r"^\s*with\b",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|copy|export)\b",
    re.IGNORECASE,
)

# Últimas mensagens de erro DuckDB (para debug / deteção de padrões novos).
_RECENT_DUCK_ERRORS: deque[str] = deque(maxlen=200)

_STRING_DATE_HEURISTIC_CACHE: dict[tuple[int, str], bool] = {}


def recent_duckdb_errors() -> list[str]:
    """Cópia das últimas mensagens DuckDB registadas (truncadas a 200 chars)."""
    return list(_RECENT_DUCK_ERRORS)


def clear_recent_duckdb_errors() -> None:
    """Limpa o buffer de erros (útil em testes)."""
    _RECENT_DUCK_ERRORS.clear()


def validate_select_only(sql: str) -> tuple[bool, str]:
    s = sql.strip()
    if not s:
        return False, "Query vazia."
    if not (_SELECT_RE.match(s) or _WITH_RE.match(s)):
        return False, "Apenas consultas SELECT (ou WITH … SELECT) são permitidas."
    if ";" in s.rstrip(";"):
        return False, "Uma única instrução SELECT por vez."
    if _FORBIDDEN.search(s):
        return False, "Palavra-chave não permitida na query."
    return True, ""


def normalize_sql_comparison_operators(sql: str) -> str:
    """Evita falhas DuckDB quando o LLM usa símbolos Unicode em comparações."""
    return sql.replace("≥", ">=").replace("≤", "<=")


def _column_name_suggests_date_string(col: str) -> bool:
    low = str(col).lower()
    return (
        "data" in low
        or "nasc" in low
        or "date" in low
        or low.endswith("_dt")
    )


def _column_is_non_datetime_stringlike(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_bool_dtype(series):
        return False
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or str(getattr(series.dtype, "name", "")).startswith("string")
    )


def _heuristic_column_parses_as_dates(
    df: pd.DataFrame,
    col: str,
    *,
    threshold: float = 0.7,
) -> bool:
    """True se a maior parte dos valores não nulos em coluna texto parecem datas."""
    key = (id(df), str(col))
    if key in _STRING_DATE_HEURISTIC_CACHE:
        return _STRING_DATE_HEURISTIC_CACHE[key]
    try:
        series = df[col]
    except (KeyError, TypeError):
        _STRING_DATE_HEURISTIC_CACHE[key] = False
        return False
    if not _column_is_non_datetime_stringlike(series):
        _STRING_DATE_HEURISTIC_CACHE[key] = False
        return False
    s = series.dropna().head(500)
    if s.empty:
        _STRING_DATE_HEURISTIC_CACHE[key] = False
        return False
    ss = s.astype(str)
    parsed = pd.to_datetime(ss, format="%Y-%m-%d", errors="coerce")
    if float(parsed.notna().mean()) <= threshold:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(ss, errors="coerce")
    ok = float(parsed.notna().mean()) > threshold
    _STRING_DATE_HEURISTIC_CACHE[key] = ok
    return ok


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if not inner:
            break
        s = inner
    return s


def _group_by_expression_is_aggregate(expr: str) -> bool:
    s = _strip_outer_parens(expr)
    return bool(
        re.match(
            r"^(AVG|SUM|COUNT|MIN|MAX|STDDEV(?:_SAMP|_POP)?|VAR(?:_SAMP|_POP)?|VARIANCE)\b\s*\(",
            s,
            re.IGNORECASE,
        )
    )


def _split_group_by_expressions(inner: str) -> list[str]:
    depth = 0
    cur: list[str] = []
    parts: list[str] = []
    i = 0
    n = len(inner)
    in_quote: str | None = None
    while i < n:
        ch = inner[i]
        if in_quote:
            if ch == "\\" and i + 1 < n:
                cur.append(ch)
                cur.append(inner[i + 1])
                i += 2
                continue
            cur.append(ch)
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in "'\"":
            in_quote = ch
            cur.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _group_by_clause_end(sql: str, start_after_by: int) -> int:
    """Índice do fim exclusivo da lista do GROUP BY (antes de ORDER/LIMIT/)/etc.).

    Usa profundidade de parêntesis **só dentro da lista** para não confundir
    `GROUP BY (a+b)` com o `)` que fecha a subquery.
    """
    i = start_after_by
    n = len(sql)
    depth = 0
    while i < n:
        c = sql[i]
        if c in "'\"":
            q = c
            i += 1
            while i < n:
                if sql[i] == "\\":
                    i = min(i + 2, n)
                    continue
                if sql[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            if depth > 0:
                depth -= 1
            else:
                return i
        elif depth == 0:
            tail = sql[i:]
            stripped = tail.lstrip()
            off = len(tail) - len(stripped)
            up = stripped[:32].upper()
            for kw in (
                "ORDER BY",
                "LIMIT",
                "HAVING",
                "FETCH",
                "UNION",
                "INTERSECT",
                "EXCEPT",
                "WINDOW",
            ):
                if up.startswith(kw) and (
                    len(stripped) == len(kw) or stripped[len(kw)] in " \n\t\r("
                ):
                    return i + off
        i += 1
    return n


def rewrite_aggregate_in_group_by(sql: str) -> str:
    """Remove expressões agregadas do GROUP BY (Binder Error «cannot contain aggregates»)."""
    out = sql
    for m in reversed(list(re.finditer(r"\bGROUP\s+BY\b", out, re.IGNORECASE))):
        kw_end = m.end()
        end_idx = _group_by_clause_end(out, kw_end)
        inner = out[kw_end:end_idx]
        parts = _split_group_by_expressions(inner)
        kept = [p for p in parts if not _group_by_expression_is_aggregate(p)]
        if len(kept) == len(parts):
            continue
        if not kept:
            # Remove o GROUP BY inteiro (ex.: só havia AVG(...) no GROUP BY).
            before = out[: m.start()].rstrip()
            after = out[end_idx:].lstrip()
            if after:
                out = f"{before} {after}".strip()
            else:
                out = before
            continue
        new_clause = " " + ", ".join(kept) + " "
        out = out[:kw_end] + new_clause + out[end_idx:]
    return out


def rewrite_date_functions_for_string_columns(sql: str, df: pd.DataFrame) -> str:
    """Envolve colunas de data em texto em TRY_CAST(... AS DATE) antes de date_part/year/EXTRACT/strftime.

    Corrige «date_part(STRING_LITERAL, VARCHAR)», year/EXTRACT/strftime sobre VARCHAR, e aplica
    heurística de parsing a colunas texto que parecem datas (>70% parseáveis).
    """
    out = sql
    for col in df.columns:
        cname = str(col)
        try:
            series = df[col]
        except (KeyError, TypeError):
            continue
        if not _column_is_non_datetime_stringlike(series):
            continue
        if not (
            _column_name_suggests_date_string(cname)
            or _heuristic_column_parses_as_dates(df, cname)
        ):
            continue
        esc = re.escape(cname)

        def _dp_repl(m: re.Match[str]) -> str:
            return f"date_part({m.group(1)}, TRY_CAST({m.group(2)} AS DATE))"

        out = re.sub(
            rf"\bdate_part\s*\(\s*((?:'[^']*')|(?:\"[^\"]*\"))\s*,\s*({esc})\s*\)",
            _dp_repl,
            out,
            flags=re.IGNORECASE,
        )

        def _yr_repl(m: re.Match[str]) -> str:
            return f"year(TRY_CAST({m.group(1)} AS DATE))"

        out = re.sub(rf"\byear\s*\(\s*({esc})\s*\)", _yr_repl, out, flags=re.IGNORECASE)

        def _ext_repl(m: re.Match[str]) -> str:
            return f"year(TRY_CAST({m.group(1)} AS DATE))"

        out = re.sub(
            rf"\bEXTRACT\s*\(\s*YEAR\s+FROM\s+({esc})\s*\)",
            _ext_repl,
            out,
            flags=re.IGNORECASE,
        )

        def _strftime_repl(m: re.Match[str]) -> str:
            fmt, colref = m.group(1), m.group(2)
            return f"strftime({fmt}, TRY_CAST({colref} AS DATE))"

        out = re.sub(
            rf"\bstrftime\s*\(\s*((?:'[^']+')|(?:\"[^\"]+\"))\s*,\s*({esc})\s*\)",
            _strftime_repl,
            out,
            flags=re.IGNORECASE,
        )
    return out


def rewrite_numeric_minus_date_cast(sql: str) -> str:
    """Converte subtrações INTEGER − DATE em subtração de anos (year/TRY_CAST).

    Cobre `Ano - TRY_CAST(...)`, literais, `EXTRACT(YEAR FROM ...) - TRY_CAST`, e
    `Ano - EXTRACT(YEAR FROM ...)`.
    """
    out = sql
    pat_lit = re.compile(
        r"\b(\d{4})\s*-\s*(TRY_CAST|CAST)\s*\(\s*(\w+)\s+AS\s+DATE\s*\)",
        re.IGNORECASE,
    )
    out = pat_lit.sub(
        lambda m: f"{m.group(1)} - year({m.group(2)}({m.group(3)} AS DATE))",
        out,
    )
    pat_col = re.compile(
        r"\b(Ano|ano_referencia)\s*-\s*(TRY_CAST|CAST)\s*\(\s*(\w+)\s+AS\s+DATE\s*\)",
        re.IGNORECASE,
    )
    out = pat_col.sub(
        lambda m: f"{m.group(1)} - year({m.group(2)}({m.group(3)} AS DATE))",
        out,
    )
    pat_ext_minus_cast = re.compile(
        r"\bEXTRACT\s*\(\s*YEAR\s+FROM\s+(\w+)\s*\)\s*-\s*(TRY_CAST|CAST)\s*\(\s*(\w+)\s+AS\s+DATE\s*\)",
        re.IGNORECASE,
    )
    out = pat_ext_minus_cast.sub(
        lambda m: (
            f"year({m.group(2)}({m.group(1)} AS DATE)) - "
            f"year({m.group(2)}({m.group(3)} AS DATE))"
        ),
        out,
    )
    pat_col_minus_extract = re.compile(
        r"\b(Ano|ano_referencia|\d{4})\s*-\s*EXTRACT\s*\(\s*YEAR\s+FROM\s+(\w+)\s*\)",
        re.IGNORECASE,
    )
    out = pat_col_minus_extract.sub(
        lambda m: f"{m.group(1)} - year(TRY_CAST({m.group(2)} AS DATE))",
        out,
    )
    return out


_COUNT_STAR_COL = re.compile(r"^count_star\(\)\s*$", re.IGNORECASE)


def prettify_sql_result_columns(out: pd.DataFrame) -> pd.DataFrame:
    """DuckDB expõe `COUNT(*)` sem alias como coluna `count_star()` — renomeia para leitura humana/UI."""
    if out is None or out.empty:
        return out
    rename: dict[str, str] = {}
    occupied = {str(c) for c in out.columns}
    for c in list(out.columns):
        if not _COUNT_STAR_COL.match(str(c).strip()):
            continue
        for cand in ("total", "n_registos", "quantidade", "contagem", "n"):
            if cand not in occupied:
                rename[str(c)] = cand
                occupied.add(cand)
                break
        else:
            i = 1
            while f"n_{i}" in occupied:
                i += 1
            nm = f"n_{i}"
            rename[str(c)] = nm
            occupied.add(nm)
    return out.rename(columns=rename) if rename else out


def apply_sql_rewrites(sql: str, df: pd.DataFrame) -> str:
    """Aplica a mesma cadeia de reescritas que `run_sql` usa antes do LIMIT/execução."""
    limited = normalize_sql_comparison_operators(sql.rstrip().rstrip(";"))
    limited = rewrite_aggregate_in_group_by(limited)
    limited = rewrite_date_functions_for_string_columns(limited, df)
    limited = rewrite_numeric_minus_date_cast(limited)
    return limited


def run_sql(
    sql: str,
    parquet_path=None,
    *,
    df: pd.DataFrame | None = None,
    bundle=None,
) -> pd.DataFrame:
    ok, err = validate_select_only(sql)
    if not ok:
        raise ValueError(err)
    if df is None:
        path = parquet_path or get_parquet_path()
        if not path.exists():
            raise FileNotFoundError(f"Parquet não encontrado: {path}")
        df = load_dados_df(path)
    # Import tardio evita ciclo ml.inference → … → data_engine.query
    from passos_magico.ml.inference import ensure_risco_column

    df = ensure_risco_column(df, bundle)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("dados", df)
        limited = apply_sql_rewrites(sql, df)
        if "limit" not in limited.lower():
            limited = f"{limited} LIMIT 5000"
        try:
            return prettify_sql_result_columns(con.execute(limited).df())
        except duckdb.Error as e:
            msg = str(e) or getattr(e, "message", "") or type(e).__name__
            _RECENT_DUCK_ERRORS.append(msg[:200])
            raise
    finally:
        con.close()
