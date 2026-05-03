"""Leitura/escrita do dicionário de dados (semantic layer)."""

from __future__ import annotations

import json
from pathlib import Path

from passos_magico.data_engine.loader import PROJECT_ROOT

DICT_PATH = PROJECT_ROOT / "dicionario.json"
RESUMO_ANUAL_PATH = PROJECT_ROOT / "resumo_anual.txt"
GAMMA_CONTEXT_PATH = PROJECT_ROOT / ".passos_gamma_context.txt"


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
            "descricao": "Probabilidade estimada pelo modelo de ML de alto risco escolar (defasagem/evasão), entre 0 e 1. Na UI, o limiar operacional de alto risco segue o treino (ex.: risco >= 0.46).",
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


def load_annual_summary_text() -> str:
    """Texto livre do resumo anual; vazio se o arquivo ainda não existir."""
    if not RESUMO_ANUAL_PATH.exists():
        return ""
    try:
        return RESUMO_ANUAL_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def load_gamma_context_text() -> str:
    """Texto institucional (ex.: export Gamma); vazio se o ficheiro não existir."""
    if not GAMMA_CONTEXT_PATH.exists():
        return ""
    try:
        return GAMMA_CONTEXT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_annual_summary_text(text: str) -> None:
    RESUMO_ANUAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = text.strip()
    RESUMO_ANUAL_PATH.write_text(body + ("\n" if body else ""), encoding="utf-8")


def merge_theo_context_blocks(
    dictionary_block: str, annual_plain: str, gamma_plain: str
) -> str:
    """Junta dicionário, resumo anual (`resumo_anual.txt`) e texto Gamma opcional para o contexto do Theo."""
    out = dictionary_block
    annual_plain = annual_plain.strip()
    gamma_plain = gamma_plain.strip()
    if annual_plain:
        out += (
            "\n\n### Resumo anual institucional (PEDE_PASSOS)\n"
            + annual_plain
            + "\n\n---\n**Uso deste bloco:** absorva prioridades, datas, conquistas e linguagem institucional para o storytelling e para interpretar os dados. "
            "Valores agregados (médias, contagens, comparações) devem vir da consulta SQL na tabela **dados**, salvo quando o gestor pedir reflexão explícita só sobre este texto."
        )
    if gamma_plain:
        out += (
            "\n\n### Contexto narrativo (Gamma / relatório anual)\n"
            + gamma_plain
            + "\n\n---\n**Uso deste bloco:** use para storytelling e alinhamento institucional; **não** trate nomes ou números deste texto como colunas da tabela **dados**. "
            "Métricas e listagens devem continuar a vir exclusivamente do SQL sobre **dados**, salvo pedido explícito de reflexão só sobre este texto."
        )
    return out


def merge_dictionary_and_annual(dictionary_block: str, annual_plain: str) -> str:
    """Compat: dicionário + resumo anual (sem bloco Gamma)."""
    return merge_theo_context_blocks(dictionary_block, annual_plain, "")
