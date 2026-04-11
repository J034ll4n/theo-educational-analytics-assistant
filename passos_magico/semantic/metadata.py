"""Leitura/escrita do dicionário de dados (semantic layer)."""

from __future__ import annotations

import json
from pathlib import Path

from passos_magico.data_engine.loader import PROJECT_ROOT

DICT_PATH = PROJECT_ROOT / "dicionario.json"


def default_dictionary_rows() -> list[dict]:
    return [
        {
            "coluna": "RA",
            "descricao": "Registro do Aluno (identificador único).",
        },
        {
            "coluna": "Nome",
            "descricao": "Nome do aluno.",
        },
        {
            "coluna": "Fase",
            "descricao": "Fase escolar no programa (ex.: 1–8).",
        },
        {
            "coluna": "Turma",
            "descricao": "Turma (ex.: A, B, C, D).",
        },
        {
            "coluna": "Ano",
            "descricao": "Ano de referência do relatório (2020–2023).",
        },
        {
            "coluna": "INDE",
            "descricao": "Índice de Desenvolvimento Educacional.",
        },
        {
            "coluna": "IDA",
            "descricao": "Indicador de Aprendizagem.",
        },
        {
            "coluna": "IAN",
            "descricao": "Indicador de Adequação ao Nível.",
        },
        {
            "coluna": "IEG",
            "descricao": "Indicador de Engajamento.",
        },
        {
            "coluna": "IPV",
            "descricao": "Indicador de Ponto de Virada.",
        },
        {
            "coluna": "Pedra",
            "descricao": "Nível Pedra: Quartzo, Ágata, Ametista ou Topázio.",
        },
        {
            "coluna": "risco",
            "descricao": "Probabilidade estimada pelo modelo de ML de alto risco escolar (defasagem/evasão), entre 0 e 1. Para contar alunos em defasagem pelo modelo, use um limiar (ex.: risco >= 0.35).",
        },
    ]


def load_dictionary(path: Path | None = None) -> list[dict]:
    p = path or DICT_PATH
    if not p.exists():
        rows = default_dictionary_rows()
        save_dictionary(rows, p)
        return rows
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("rows", default_dictionary_rows())


def save_dictionary(rows: list[dict], path: Path | None = None) -> None:
    p = path or DICT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def rows_to_prompt_block(rows: list[dict]) -> str:
    lines = ["### Dicionário de colunas (PEDE_PASSOS)"]
    for r in rows:
        col = r.get("coluna", "")
        desc = r.get("descricao", "")
        lines.append(f"- **{col}**: {desc}")
    return "\n".join(lines)
