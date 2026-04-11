"""Extrai SQL de resposta do modelo."""

from __future__ import annotations

import re


def extract_sql_block(text: str) -> str | None:
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    t = text.strip()
    if re.match(r"^\s*select\b", t, re.IGNORECASE):
        return t.rstrip(";")
    return None


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
    return [
        "Comparar o desempenho entre turmas no mesmo ano?",
        "Ver a distribuição de Pedras por fase?",
        "Evolução do INDE ao longo dos anos?",
    ]
