"""Motor DuckDB sobre Parquet — apenas SELECT."""

from __future__ import annotations

import re

import duckdb
import pandas as pd

from passos_magico.data_engine.loader import get_parquet_path

_SELECT_RE = re.compile(
    r"^\s*select\b",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|copy|export)\b",
    re.IGNORECASE,
)


def validate_select_only(sql: str) -> tuple[bool, str]:
    s = sql.strip()
    if not s:
        return False, "Query vazia."
    if not _SELECT_RE.match(s):
        return False, "Apenas consultas SELECT são permitidas."
    if ";" in s.rstrip(";"):
        return False, "Uma única instrução SELECT por vez."
    if _FORBIDDEN.search(s):
        return False, "Palavra-chave não permitida na query."
    return True, ""


def run_sql(sql: str, parquet_path=None) -> pd.DataFrame:
    ok, err = validate_select_only(sql)
    if not ok:
        raise ValueError(err)
    path = parquet_path or get_parquet_path()
    if not path.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {path}")
    p = str(path).replace("'", "''")
    # Tabela fixa 'dados' para o LLM
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"CREATE VIEW dados AS SELECT * FROM read_parquet('{p}');")
        limited = sql.rstrip().rstrip(";")
        if "limit" not in limited.lower():
            limited = f"{limited} LIMIT 5000"
        return con.execute(limited).df()
    finally:
        con.close()
