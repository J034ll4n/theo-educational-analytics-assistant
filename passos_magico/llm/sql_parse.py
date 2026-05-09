"""Extrai SQL de resposta do modelo."""

from __future__ import annotations

import re
from collections.abc import Iterable


_SQL_START = re.compile(r"^\s*(?:with|select)\b", re.IGNORECASE)


def extract_sql_block(text: str) -> str | None:
    """Extrai SQL de markdown ```sql``` ou texto solto (SELECT ou WITH … SELECT)."""
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()

    # 1) Blocos ```sql``` ou ``` … ``` — CTEs começam com WITH, não SELECT
    for m in re.finditer(r"```(?:sql)?\s*([\s\S]*?)```", raw, re.IGNORECASE):
        inner = m.group(1).strip()
        if _SQL_START.match(inner):
            return _clean_sql(inner)

    # 2) Texto começa direto com SELECT ou WITH
    if _SQL_START.match(raw):
        return _clean_sql(raw)

    # 3) SELECT ou WITH no meio do texto (prosa antes) — usa o primeiro WITH ou SELECT “de topo”
    mw = re.search(r"\bWITH\b", raw, re.IGNORECASE)
    ms = re.search(r"\bSELECT\b", raw, re.IGNORECASE)
    if mw and (ms is None or mw.start() <= ms.start()):
        pos = mw.start()
    elif ms:
        pos = ms.start()
    else:
        return None

    tail = raw[pos:].strip()
    if ";" in tail:
        stmt, rest = tail.split(";", 1)
        stmt = stmt.strip()
        if re.search(r"\bfrom\b", stmt, re.IGNORECASE):
            return _clean_sql(stmt)
    elif re.search(r"\bfrom\b", tail, re.IGNORECASE):
        chunk = re.split(r"\n\s*\n\s*\n", tail, maxsplit=1)[0].strip()
        return _clean_sql(chunk)

    return None


def _clean_sql(s: str) -> str:
    s = s.strip().rstrip(";").strip("`").strip()
    if s.endswith("```"):
        s = s[: s.rfind("```")].strip()
    return s.strip()


# Referência à view `dados`: não citada, "dados" (DuckDB) ou `dados`
_FROM_DATOS_RE = re.compile(
    r'\bfrom\s+(?:"dados"|`dados`|\bdados\b)',
    re.IGNORECASE,
)


def sql_passes_quick_validation(sql: str | None) -> bool:
    """Evita executar SQL óbvio demais: referência a `dados`; parênteses só como aviso leve."""
    if not sql:
        return False
    s = sql.strip()
    if len(s) < 14:
        return False
    # Parênteses muito desiguais costumam ser extração truncada; pequenas diferenças deixamos para o DuckDB
    if abs(s.count("(") - s.count(")")) > 2:
        return False
    if not _FROM_DATOS_RE.search(s):
        return False
    # CTE quebrado: primeira linha termina em "(" e a segunda é só ")"
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 2 and lines[1] in (")", ");"):
        if re.search(r"\(\s*$", lines[0]):
            return False
    return True


SUGGESTION_FALLBACK: tuple[str, ...] = (
    "Comparar o desempenho entre turmas no mesmo ano?",
    "Ver a distribuição de Pedras por fase?",
    "Evolução do INDE ao longo dos anos?",
)


def extract_json_suggestions(text: str) -> list[str]:
    import json

    text = text.strip()
    # Remove cercas markdown ```json ... ``` que o modelo às vezes envolve na resposta
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        text = text.rstrip()
        if text.endswith("```"):
            text = text[: text.rfind("```")].rstrip()
    text = text.strip()
    try:
        # tenta JSON puro
        data = json.loads(text)
        if isinstance(data, dict) and "sugestoes" in data:
            return list(data["sugestoes"])[:3]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "sugestoes" in data:
                return list(data["sugestoes"])[:3]
        except json.JSONDecodeError:
            pass
    return list(SUGGESTION_FALLBACK)


# Tokens que o LLM costuma inventar como coluna; bloquear cedo se não existirem no Parquet.
_DISALLOWED_SQL_TOKENS = frozenset(
    {
        "feedback",
        "comentario",
        "comentarios",
        "avaliacao",
        "resumo_anual",
    }
)


def _sql_without_string_literals(sql: str) -> str:
    """Remove literais '…' e \"…\" (ingénuo, suficiente para o guard)."""
    s = sql
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in "'\"":
            quote = ch
            out.append(" ")
            i += 1
            while i < n:
                if s[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if s[i] == quote:
                    if quote == "'" and i + 1 < n and s[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def sql_guard_disallowed_tokens(sql: str, known_columns: Iterable[str] | None) -> tuple[bool, str]:
    """
    Rejeita SQL que referencia tokens sabidamente alucinados se não existirem nas colunas reais.
    Conservador: só lista fechada em _DISALLOWED_SQL_TOKENS.
    """
    if not known_columns:
        return True, ""
    known = {str(c).lower() for c in known_columns}
    cleaned = _sql_without_string_literals(sql).lower()
    for tok in _DISALLOWED_SQL_TOKENS:
        if tok in known:
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(tok)}(?![a-z0-9_])", cleaned):
            return False, f"Token `{tok}` não corresponde a coluna na base carregada."
    return True, ""
