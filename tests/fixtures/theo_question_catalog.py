"""Catálogo de 100 perguntas + SQL ouro para o Theo — usado em testes e na UI (exemplos).

Todas as consultas assumem a view `dados` com o esquema devolvido por `build_catalog_demo_df()`.

`CATEGORIA_PESO_RELATIVA` define a probabilidade relativa esperada por tipo de pergunta (soma 1);
use `amostra_perguntas_por_peso` para sugerir perguntas alinhadas a essa distribuição.
"""

from __future__ import annotations

import random
from typing import TypedDict

import pandas as pd


class QuestionCase(TypedDict):
    id: str
    categoria: str
    pergunta: str
    sql_ouro: str
    esperado_cols: list[str]


# Probabilidade relativa esperada por categoria (soma = 1.0), alinhada ao nº de perguntas no catálogo (100).
# Reflete frequência típica: mais contagens/médias/evolução e risco; menos CTEs muito elaboradas.
CATEGORIA_PESO_RELATIVA: dict[str, float] = {
    "basicas": 0.14,
    "medias_dimensao": 0.16,
    "evolucao_temporal": 0.11,
    "comparacoes_rede": 0.07,
    "risco_modelo": 0.13,
    "idade_data": 0.14,
    "multi_indicador": 0.10,
    "ranking_topn": 0.08,
    "subset_complexo": 0.07,
}


def categoria_probabilidades() -> dict[str, float]:
    """Cópia imutável das probabilidades por categoria (para UI, docs ou amostragem)."""
    return dict(CATEGORIA_PESO_RELATIVA)


def _n_por_categoria(cat: str) -> int:
    return sum(1 for c in QUESTION_CATALOG if c["categoria"] == cat)


def peso_amostragem_pergunta(case: QuestionCase) -> float:
    """Peso de uma pergunta na amostragem: P(categoria) / N perguntas nessa categoria."""
    base = CATEGORIA_PESO_RELATIVA.get(case["categoria"], 0.0)
    n = _n_por_categoria(case["categoria"])
    return base / n if n else 0.0


def amostra_perguntas_por_peso(*, k: int = 10, seed: int | None = None) -> list[str]:
    """Devolve até `k` perguntas (texto) com reposição, respeitando `CATEGORIA_PESO_RELATIVA`."""
    if not QUESTION_CATALOG or k <= 0:
        return []
    rng = random.Random(seed)
    perguntas = [c["pergunta"] for c in QUESTION_CATALOG]
    pesos = [peso_amostragem_pergunta(c) for c in QUESTION_CATALOG]
    s = sum(pesos)
    if s <= 0:
        return rng.sample(perguntas, k=min(k, len(perguntas)))
    w = [p / s for p in pesos]
    return rng.choices(perguntas, weights=w, k=k)


def build_catalog_demo_df() -> pd.DataFrame:
    """DataFrame mínimo mas rico o suficiente para todas as entradas do catálogo."""
    rows: list[dict[str, object]] = []
    # Mesmos RA em vários anos (evolução / mudança de pedra)
    ra_special = "RA-MUD"
    for ano, pedra, ida in [
        (2022, "Quartzo", 5.0),
        (2023, "Ágata", 5.8),
        (2024, "Ametista", 6.2),
    ]:
        rows.append(
            {
                "RA": ra_special,
                "Nome": "Aluno Mudança",
                "Fase": 8,
                "Turma": "A",
                "Ano": ano,
                "INDE": 6.0 + (ano - 2022) * 0.3,
                "IDA": ida,
                "IAN": 6.5,
                "IEG": 6.0,
                "IPV": 5.5 + (ano - 2022) * 0.2,
                "Pedra": pedra,
                "risco": 0.35 + (ano - 2022) * 0.05,
                "data_nasc": "2006-04-15",
                "escola_publica": True,
                "ano_ingresso": 2020,
                "defasado": False,
            }
        )
    # Base genérica
    for i in range(24):
        fase = 6 + (i % 3)
        turma = ["A", "B", "C"][i % 3]
        ano = [2021, 2022, 2023, 2024][i % 4]
        pub = i % 3 != 0
        birth = f"{2004 + (i % 8):04d}-{(i % 12) + 1:02d}-15"
        rows.append(
            {
                "RA": f"RA-{i:03d}",
                "Nome": f"Aluno {i}",
                "Fase": fase,
                "Turma": turma,
                "Ano": ano,
                "INDE": 5.5 + (i % 5) * 0.4,
                "IDA": 4.5 + (i % 6) * 0.35,
                "IAN": 5.8 + (i % 4) * 0.25,
                "IEG": 5.2 + (i % 5) * 0.3,
                "IPV": 5.0 + (i % 7) * 0.2,
                "Pedra": ["Quartzo", "Ágata", "Ametista", "Topázio"][i % 4],
                "risco": 0.2 + (i % 10) * 0.07,
                "data_nasc": birth,
                "escola_publica": pub,
                "ano_ingresso": 2018 + (i % 5),
                "defasado": i % 7 == 0,
            }
        )
    df = pd.DataFrame(rows)
    df["data_nasc"] = df["data_nasc"].astype("string")
    return df


def theo_test_question_groups(max_per_categoria: int = 4) -> tuple[tuple[str, list[str]], ...]:
    """Agrupa perguntas por categoria para o expander do Streamlit."""
    grupos: dict[str, list[str]] = {}
    for case in QUESTION_CATALOG:
        grupos.setdefault(case["categoria"], []).append(case["pergunta"])
    out: list[tuple[str, list[str]]] = []
    for cat in sorted(grupos.keys()):
        items = grupos[cat][:max_per_categoria]
        if items:
            label = {
                "basicas": "Básicas",
                "medias_dimensao": "Médias por dimensão",
                "evolucao_temporal": "Evolução temporal",
                "comparacoes_rede": "Pública vs particular",
                "risco_modelo": "Risco (modelo)",
                "idade_data": "Idade e datas",
                "multi_indicador": "Vários indicadores",
                "ranking_topn": "Ranking / Top N",
                "subset_complexo": "Consultas complexas (CTE)",
            }.get(cat, cat.replace("_", " ").title())
            out.append((label, items))
    return tuple(out)


QUESTION_CATALOG: list[QuestionCase] = [
    # --- basicas (6) ---
    {
        "id": "b1",
        "categoria": "basicas",
        "pergunta": "Quantos registros existem na base no total?",
        "sql_ouro": "SELECT COUNT(*) AS total FROM dados",
        "esperado_cols": ["total"],
    },
    {
        "id": "b2",
        "categoria": "basicas",
        "pergunta": "Quantos alunos únicos (RA distintos) existem?",
        "sql_ouro": "SELECT COUNT(DISTINCT RA) AS n_alunos FROM dados",
        "esperado_cols": ["n_alunos"],
    },
    {
        "id": "b3",
        "categoria": "basicas",
        "pergunta": "Liste as turmas distintas na Fase 8.",
        "sql_ouro": "SELECT DISTINCT Turma FROM dados WHERE Fase = 8 ORDER BY Turma",
        "esperado_cols": ["Turma"],
    },
    {
        "id": "b4",
        "categoria": "basicas",
        "pergunta": "Qual o valor médio de INDE considerando todos os registros?",
        "sql_ouro": "SELECT AVG(INDE) AS media_inde FROM dados",
        "esperado_cols": ["media_inde"],
    },
    {
        "id": "b5",
        "categoria": "basicas",
        "pergunta": "Quantos registros há para cada Ano?",
        "sql_ouro": "SELECT Ano, COUNT(*) AS n FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "n"],
    },
    {
        "id": "b6",
        "categoria": "basicas",
        "pergunta": "Quantos alunos estão com defasado = true?",
        "sql_ouro": "SELECT COUNT(*) AS total_defasados FROM dados WHERE defasado = true",
        "esperado_cols": ["total_defasados"],
    },
    {
        "id": "b7",
        "categoria": "basicas",
        "pergunta": "Quantos registros existem com Ano = 2022?",
        "sql_ouro": "SELECT COUNT(*) AS n FROM dados WHERE Ano = 2022",
        "esperado_cols": ["n"],
    },
    {
        "id": "b8",
        "categoria": "basicas",
        "pergunta": "Quantos registros na Fase 6?",
        "sql_ouro": "SELECT COUNT(*) AS n FROM dados WHERE Fase = 6",
        "esperado_cols": ["n"],
    },
    {
        "id": "b9",
        "categoria": "basicas",
        "pergunta": "Liste os níveis de Pedra distintos na base.",
        "sql_ouro": "SELECT DISTINCT Pedra FROM dados ORDER BY Pedra",
        "esperado_cols": ["Pedra"],
    },
    {
        "id": "b10",
        "categoria": "basicas",
        "pergunta": "Quantos alunos não estão defasados (defasado = false)?",
        "sql_ouro": "SELECT COUNT(*) AS n FROM dados WHERE defasado = false",
        "esperado_cols": ["n"],
    },
    {
        "id": "b11",
        "categoria": "basicas",
        "pergunta": "Qual o ano mínimo e o ano máximo presentes na base?",
        "sql_ouro": "SELECT MIN(Ano) AS ano_min, MAX(Ano) AS ano_max FROM dados",
        "esperado_cols": ["ano_min", "ano_max"],
    },
    {
        "id": "b12",
        "categoria": "basicas",
        "pergunta": "Qual a média global de IDA em todos os registros?",
        "sql_ouro": "SELECT AVG(IDA) AS media_ida FROM dados",
        "esperado_cols": ["media_ida"],
    },
    {
        "id": "b13",
        "categoria": "basicas",
        "pergunta": "Quantos registros são de escola pública?",
        "sql_ouro": "SELECT COUNT(*) AS n FROM dados WHERE escola_publica = true OR escola_publica = 1",
        "esperado_cols": ["n"],
    },
    {
        "id": "b14",
        "categoria": "basicas",
        "pergunta": "Liste as Fases distintas presentes nos dados.",
        "sql_ouro": "SELECT DISTINCT Fase FROM dados ORDER BY Fase",
        "esperado_cols": ["Fase"],
    },
    # --- medias_dimensao (16) ---
    {
        "id": "m1",
        "categoria": "medias_dimensao",
        "pergunta": "Qual a média de IDA por Fase em 2022?",
        "sql_ouro": "SELECT Fase, AVG(IDA) AS media_ida FROM dados WHERE Ano = 2022 GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "media_ida"],
    },
    {
        "id": "m2",
        "categoria": "medias_dimensao",
        "pergunta": "Média de INDE por Turma no ano 2023.",
        "sql_ouro": "SELECT Turma, AVG(INDE) AS media_inde FROM dados WHERE Ano = 2023 GROUP BY Turma ORDER BY Turma",
        "esperado_cols": ["Turma", "media_inde"],
    },
    {
        "id": "m3",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IAN ano a ano (todos os anos da base).",
        "sql_ouro": "SELECT Ano, AVG(IAN) AS media_ian FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "media_ian"],
    },
    {
        "id": "m4",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IEG por Pedra em 2024.",
        "sql_ouro": "SELECT Pedra, AVG(IEG) AS media_ieg FROM dados WHERE Ano = 2024 GROUP BY Pedra ORDER BY Pedra",
        "esperado_cols": ["Pedra", "media_ieg"],
    },
    {
        "id": "m5",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IPV por Fase e Turma em 2022.",
        "sql_ouro": "SELECT Fase, Turma, AVG(IPV) AS media_ipv FROM dados WHERE Ano = 2022 GROUP BY Fase, Turma ORDER BY Fase, Turma",
        "esperado_cols": ["Fase", "Turma", "media_ipv"],
    },
    {
        "id": "m6",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IDA na Fase 7 turma B em 2021.",
        "sql_ouro": "SELECT AVG(IDA) AS media_ida FROM dados WHERE Ano = 2021 AND Fase = 7 AND Turma = 'B'",
        "esperado_cols": ["media_ida"],
    },
    {
        "id": "m7",
        "categoria": "medias_dimensao",
        "pergunta": "Média de INDE por Fase em 2021.",
        "sql_ouro": "SELECT Fase, AVG(INDE) AS media_inde FROM dados WHERE Ano = 2021 GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "media_inde"],
    },
    {
        "id": "m8",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IDA por Ano (todos os anos).",
        "sql_ouro": "SELECT Ano, AVG(IDA) AS media_ida FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "media_ida"],
    },
    {
        "id": "m9",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IPV por Fase em 2024.",
        "sql_ouro": "SELECT Fase, AVG(IPV) AS media_ipv FROM dados WHERE Ano = 2024 GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "media_ipv"],
    },
    {
        "id": "m10",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IAN por Turma em 2022.",
        "sql_ouro": "SELECT Turma, AVG(IAN) AS media_ian FROM dados WHERE Ano = 2022 GROUP BY Turma ORDER BY Turma",
        "esperado_cols": ["Turma", "media_ian"],
    },
    {
        "id": "m11",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IEG na Fase 6 em 2022.",
        "sql_ouro": "SELECT AVG(IEG) AS media_ieg FROM dados WHERE Fase = 6 AND Ano = 2022",
        "esperado_cols": ["media_ieg"],
    },
    {
        "id": "m12",
        "categoria": "medias_dimensao",
        "pergunta": "Média de INDE por Pedra em 2023.",
        "sql_ouro": "SELECT Pedra, AVG(INDE) AS media_inde FROM dados WHERE Ano = 2023 GROUP BY Pedra ORDER BY Pedra",
        "esperado_cols": ["Pedra", "media_inde"],
    },
    {
        "id": "m13",
        "categoria": "medias_dimensao",
        "pergunta": "Em 2024, média de INDE e de IDA por Fase.",
        "sql_ouro": "SELECT Fase, AVG(INDE) AS media_inde, AVG(IDA) AS media_ida FROM dados WHERE Ano = 2024 GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "media_inde", "media_ida"],
    },
    {
        "id": "m14",
        "categoria": "medias_dimensao",
        "pergunta": "Média de risco por Turma em 2024.",
        "sql_ouro": "SELECT Turma, AVG(risco) AS media_risco FROM dados WHERE Ano = 2024 GROUP BY Turma ORDER BY Turma",
        "esperado_cols": ["Turma", "media_risco"],
    },
    {
        "id": "m15",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IPV na Turma A em 2023.",
        "sql_ouro": "SELECT AVG(IPV) AS media_ipv FROM dados WHERE Ano = 2023 AND Turma = 'A'",
        "esperado_cols": ["media_ipv"],
    },
    {
        "id": "m16",
        "categoria": "medias_dimensao",
        "pergunta": "Média de IAN por Pedra em 2022.",
        "sql_ouro": "SELECT Pedra, AVG(IAN) AS media_ian FROM dados WHERE Ano = 2022 GROUP BY Pedra ORDER BY Pedra",
        "esperado_cols": ["Pedra", "media_ian"],
    },
    # --- evolucao_temporal (11) ---
    {
        "id": "e1",
        "categoria": "evolucao_temporal",
        "pergunta": "Evolução da média de INDE por Ano.",
        "sql_ouro": "SELECT Ano, AVG(INDE) AS media_inde FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "media_inde"],
    },
    {
        "id": "e2",
        "categoria": "evolucao_temporal",
        "pergunta": "Contagem de registros por Ano ordenada.",
        "sql_ouro": "SELECT Ano, COUNT(*) AS n FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "n"],
    },
    {
        "id": "e3",
        "categoria": "evolucao_temporal",
        "pergunta": "Para o RA-MUD, como evolui o INDE ao longo dos anos?",
        "sql_ouro": "SELECT Ano, INDE FROM dados WHERE RA = 'RA-MUD' ORDER BY Ano",
        "esperado_cols": ["Ano", "INDE"],
    },
    {
        "id": "e4",
        "categoria": "evolucao_temporal",
        "pergunta": "Média de risco do modelo por Ano.",
        "sql_ouro": "SELECT Ano, AVG(risco) AS media_risco FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "media_risco"],
    },
    {
        "id": "e5",
        "categoria": "evolucao_temporal",
        "pergunta": "Diferença entre média de IDA em 2023 e 2024 (agregado único).",
        "sql_ouro": """
SELECT AVG(CASE WHEN Ano = 2024 THEN IDA END) - AVG(CASE WHEN Ano = 2023 THEN IDA END) AS diff_ida
FROM dados
""".strip(),
        "esperado_cols": ["diff_ida"],
    },
    {
        "id": "e6",
        "categoria": "evolucao_temporal",
        "pergunta": "Evolução da média de IDA por Ano.",
        "sql_ouro": "SELECT Ano, AVG(IDA) AS media_ida FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "media_ida"],
    },
    {
        "id": "e7",
        "categoria": "evolucao_temporal",
        "pergunta": "Para RA-MUD, evolução do IDA ao longo dos anos.",
        "sql_ouro": "SELECT Ano, IDA FROM dados WHERE RA = 'RA-MUD' ORDER BY Ano",
        "esperado_cols": ["Ano", "IDA"],
    },
    {
        "id": "e8",
        "categoria": "evolucao_temporal",
        "pergunta": "Maior INDE registado por Ano (máximo agregado).",
        "sql_ouro": "SELECT Ano, MAX(INDE) AS inde_max FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "inde_max"],
    },
    {
        "id": "e9",
        "categoria": "evolucao_temporal",
        "pergunta": "Média de IEG por Ano.",
        "sql_ouro": "SELECT Ano, AVG(IEG) AS media_ieg FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "media_ieg"],
    },
    {
        "id": "e10",
        "categoria": "evolucao_temporal",
        "pergunta": "Diferença entre média de INDE em 2022 e em 2021 (um valor).",
        "sql_ouro": """
SELECT AVG(CASE WHEN Ano = 2022 THEN INDE END) - AVG(CASE WHEN Ano = 2021 THEN INDE END) AS diff_inde
FROM dados
""".strip(),
        "esperado_cols": ["diff_inde"],
    },
    {
        "id": "e11",
        "categoria": "evolucao_temporal",
        "pergunta": "Contagem de registros por Ano e por Fase.",
        "sql_ouro": "SELECT Ano, Fase, COUNT(*) AS n FROM dados GROUP BY Ano, Fase ORDER BY Ano, Fase",
        "esperado_cols": ["Ano", "Fase", "n"],
    },
    # --- comparacoes_rede (7) ---
    {
        "id": "c1",
        "categoria": "comparacoes_rede",
        "pergunta": "Quantos registros são de escola pública vs particular?",
        "sql_ouro": """SELECT CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
       COUNT(*) AS quantidade FROM dados GROUP BY 1 ORDER BY tipo_rede""",
        "esperado_cols": ["tipo_rede", "quantidade"],
    },
    {
        "id": "c2",
        "categoria": "comparacoes_rede",
        "pergunta": "Média de IDA em 2022: pública vs particular.",
        "sql_ouro": """SELECT CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
       AVG(IDA) AS media_ida FROM dados WHERE Ano = 2022 GROUP BY 1 ORDER BY tipo_rede""",
        "esperado_cols": ["tipo_rede", "media_ida"],
    },
    {
        "id": "c3",
        "categoria": "comparacoes_rede",
        "pergunta": "Média de INDE por tipo de rede no último ano (2024).",
        "sql_ouro": """SELECT CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
       AVG(INDE) AS media_inde FROM dados WHERE Ano = 2024 GROUP BY 1 ORDER BY tipo_rede""",
        "esperado_cols": ["tipo_rede", "media_inde"],
    },
    {
        "id": "c4",
        "categoria": "comparacoes_rede",
        "pergunta": "Contagem por tipo de rede na Fase 8.",
        "sql_ouro": """SELECT CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
       COUNT(*) AS n FROM dados WHERE Fase = 8 GROUP BY 1 ORDER BY tipo_rede""",
        "esperado_cols": ["tipo_rede", "n"],
    },
    {
        "id": "c5",
        "categoria": "comparacoes_rede",
        "pergunta": "Média de IAN em 2023: escola pública vs particular.",
        "sql_ouro": """SELECT CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
       AVG(IAN) AS media_ian FROM dados WHERE Ano = 2023 GROUP BY 1 ORDER BY tipo_rede""",
        "esperado_cols": ["tipo_rede", "media_ian"],
    },
    {
        "id": "c6",
        "categoria": "comparacoes_rede",
        "pergunta": "Contagem por tipo de rede no ano 2021.",
        "sql_ouro": """SELECT CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
       COUNT(*) AS n FROM dados WHERE Ano = 2021 GROUP BY 1 ORDER BY tipo_rede""",
        "esperado_cols": ["tipo_rede", "n"],
    },
    {
        "id": "c7",
        "categoria": "comparacoes_rede",
        "pergunta": "Média de IEG por tipo de rede em 2022.",
        "sql_ouro": """SELECT CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
       AVG(IEG) AS media_ieg FROM dados WHERE Ano = 2022 GROUP BY 1 ORDER BY tipo_rede""",
        "esperado_cols": ["tipo_rede", "media_ieg"],
    },
    # --- risco_modelo (13) ---
    {
        "id": "r1",
        "categoria": "risco_modelo",
        "pergunta": "Quantos alunos têm risco do modelo >= 0.46?",
        "sql_ouro": "SELECT COUNT(*) AS n FROM dados WHERE risco >= 0.46",
        "esperado_cols": ["n"],
    },
    {
        "id": "r2",
        "categoria": "risco_modelo",
        "pergunta": "Média de risco por Fase em 2023.",
        "sql_ouro": "SELECT Fase, AVG(risco) AS media_risco FROM dados WHERE Ano = 2023 GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "media_risco"],
    },
    {
        "id": "r3",
        "categoria": "risco_modelo",
        "pergunta": "Distribuição por Turma entre alunos com risco >= 0.5 na Fase 8.",
        "sql_ouro": "SELECT Turma, COUNT(*) AS n FROM dados WHERE Fase = 8 AND risco >= 0.5 GROUP BY Turma ORDER BY n DESC",
        "esperado_cols": ["Turma", "n"],
    },
    {
        "id": "r4",
        "categoria": "risco_modelo",
        "pergunta": "Média de IDA entre quem tem risco >= 0.46 em 2024.",
        "sql_ouro": "SELECT AVG(IDA) AS media_ida FROM dados WHERE Ano = 2024 AND risco >= 0.46",
        "esperado_cols": ["media_ida"],
    },
    {
        "id": "r5",
        "categoria": "risco_modelo",
        "pergunta": "Top 3 RA com maior risco em 2024.",
        "sql_ouro": "SELECT RA, MAX(risco) AS risco_max FROM dados WHERE Ano = 2024 GROUP BY RA ORDER BY risco_max DESC LIMIT 3",
        "esperado_cols": ["RA", "risco_max"],
    },
    {
        "id": "r6",
        "categoria": "risco_modelo",
        "pergunta": "Quantos registros com risco entre 0.3 e 0.6?",
        "sql_ouro": "SELECT COUNT(*) AS n FROM dados WHERE risco >= 0.3 AND risco <= 0.6",
        "esperado_cols": ["n"],
    },
    {
        "id": "r7",
        "categoria": "risco_modelo",
        "pergunta": "Quantos alunos têm risco do modelo abaixo de 0.3?",
        "sql_ouro": "SELECT COUNT(*) AS n FROM dados WHERE risco < 0.3",
        "esperado_cols": ["n"],
    },
    {
        "id": "r8",
        "categoria": "risco_modelo",
        "pergunta": "Média de risco por Pedra em 2024.",
        "sql_ouro": "SELECT Pedra, AVG(risco) AS media_risco FROM dados WHERE Ano = 2024 GROUP BY Pedra ORDER BY Pedra",
        "esperado_cols": ["Pedra", "media_risco"],
    },
    {
        "id": "r9",
        "categoria": "risco_modelo",
        "pergunta": "Top 5 RA com maior risco em 2023.",
        "sql_ouro": "SELECT RA, MAX(risco) AS risco_max FROM dados WHERE Ano = 2023 GROUP BY RA ORDER BY risco_max DESC LIMIT 5",
        "esperado_cols": ["RA", "risco_max"],
    },
    {
        "id": "r10",
        "categoria": "risco_modelo",
        "pergunta": "Média de INDE entre quem tem risco >= 0.5 em 2022.",
        "sql_ouro": "SELECT AVG(INDE) AS media_inde FROM dados WHERE Ano = 2022 AND risco >= 0.5",
        "esperado_cols": ["media_inde"],
    },
    {
        "id": "r11",
        "categoria": "risco_modelo",
        "pergunta": "Contagem por Fase entre alunos com risco >= 0.46 em 2024.",
        "sql_ouro": "SELECT Fase, COUNT(*) AS n FROM dados WHERE Ano = 2024 AND risco >= 0.46 GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "n"],
    },
    {
        "id": "r12",
        "categoria": "risco_modelo",
        "pergunta": "Valor mínimo e máximo de risco em 2024.",
        "sql_ouro": "SELECT MIN(risco) AS risco_min, MAX(risco) AS risco_max FROM dados WHERE Ano = 2024",
        "esperado_cols": ["risco_min", "risco_max"],
    },
    {
        "id": "r13",
        "categoria": "risco_modelo",
        "pergunta": "Média de IDA na Turma B entre quem tem risco >= 0.4 em 2023.",
        "sql_ouro": "SELECT AVG(IDA) AS media_ida FROM dados WHERE Ano = 2023 AND Turma = 'B' AND risco >= 0.4",
        "esperado_cols": ["media_ida"],
    },
    # --- idade_data (14) ---
    {
        "id": "i1",
        "categoria": "idade_data",
        "pergunta": "Idade média dos alunos em anos (Ano − ano de nascimento).",
        "sql_ouro": "SELECT AVG(Ano - year(TRY_CAST(data_nasc AS DATE))) AS media_idade_anos FROM dados",
        "esperado_cols": ["media_idade_anos"],
    },
    {
        "id": "i2",
        "categoria": "idade_data",
        "pergunta": "Média do ano de nascimento (a partir de data_nasc).",
        "sql_ouro": "SELECT AVG(date_part('year', TRY_CAST(data_nasc AS DATE))) AS media_ano_nasc FROM dados",
        "esperado_cols": ["media_ano_nasc"],
    },
    {
        "id": "i3",
        "categoria": "idade_data",
        "pergunta": "Qual o impacto da idade sobre o desempenho (IDA médio por idade em anos)?",
        "sql_ouro": """SELECT (Ano - year(TRY_CAST(data_nasc AS DATE))) AS idade_anos,
       AVG(IDA) AS media_ida,
       COUNT(*) AS n
FROM dados
WHERE TRY_CAST(data_nasc AS DATE) IS NOT NULL
GROUP BY 1
ORDER BY 1
LIMIT 120""",
        "esperado_cols": ["idade_anos", "media_ida", "n"],
    },
    {
        "id": "i4",
        "categoria": "idade_data",
        "pergunta": "IDA médio por faixa etária simplificada.",
        "sql_ouro": """SELECT CASE
         WHEN (Ano - year(TRY_CAST(data_nasc AS DATE))) <= 15 THEN 'até 15'
         ELSE '16+'
       END AS faixa_idade,
       AVG(IDA) AS media_ida,
       COUNT(*) AS n
FROM dados
WHERE TRY_CAST(data_nasc AS DATE) IS NOT NULL
GROUP BY 1
ORDER BY 1""",
        "esperado_cols": ["faixa_idade", "media_ida", "n"],
    },
    {
        "id": "i5",
        "categoria": "idade_data",
        "pergunta": "Média de ano de ingresso por Fase.",
        "sql_ouro": "SELECT Fase, AVG(ano_ingresso) AS media_ingresso FROM dados GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "media_ingresso"],
    },
    {
        "id": "i6",
        "categoria": "idade_data",
        "pergunta": "Contagem de alunos com data de nascimento válida em 2024.",
        "sql_ouro": "SELECT COUNT(*) AS n FROM dados WHERE Ano = 2024 AND TRY_CAST(data_nasc AS DATE) IS NOT NULL",
        "esperado_cols": ["n"],
    },
    {
        "id": "i7",
        "categoria": "idade_data",
        "pergunta": "Média de IAN por idade em anos (arredondada) em 2023.",
        "sql_ouro": """SELECT ROUND(Ano - year(TRY_CAST(data_nasc AS DATE))) AS idade_arred,
       AVG(IAN) AS media_ian
FROM dados WHERE Ano = 2023
GROUP BY 1
ORDER BY 1""",
        "esperado_cols": ["idade_arred", "media_ian"],
    },
    {
        "id": "i8",
        "categoria": "idade_data",
        "pergunta": "Mínimo e máximo de idade em anos na base (em 2024).",
        "sql_ouro": """SELECT MIN(Ano - year(TRY_CAST(data_nasc AS DATE))) AS idade_min,
       MAX(Ano - year(TRY_CAST(data_nasc AS DATE))) AS idade_max
FROM dados WHERE Ano = 2024""",
        "esperado_cols": ["idade_min", "idade_max"],
    },
    {
        "id": "i9",
        "categoria": "idade_data",
        "pergunta": "Idade média em anos apenas para registos de 2022.",
        "sql_ouro": "SELECT AVG(Ano - year(TRY_CAST(data_nasc AS DATE))) AS media_idade_anos FROM dados WHERE Ano = 2022",
        "esperado_cols": ["media_idade_anos"],
    },
    {
        "id": "i10",
        "categoria": "idade_data",
        "pergunta": "Quantos registos em 2024 têm idade em anos estritamente maior que 12?",
        "sql_ouro": """SELECT COUNT(*) AS n FROM dados WHERE Ano = 2024
AND TRY_CAST(data_nasc AS DATE) IS NOT NULL
AND (Ano - year(TRY_CAST(data_nasc AS DATE))) > 12""",
        "esperado_cols": ["n"],
    },
    {
        "id": "i11",
        "categoria": "idade_data",
        "pergunta": "Média de INDE por idade em anos (2023, até 80 linhas).",
        "sql_ouro": """SELECT (Ano - year(TRY_CAST(data_nasc AS DATE))) AS idade_anos,
       AVG(INDE) AS media_inde,
       COUNT(*) AS n
FROM dados WHERE Ano = 2023 AND TRY_CAST(data_nasc AS DATE) IS NOT NULL
GROUP BY 1
ORDER BY 1
LIMIT 80""",
        "esperado_cols": ["idade_anos", "media_inde", "n"],
    },
    {
        "id": "i12",
        "categoria": "idade_data",
        "pergunta": "Média de ano de ingresso por Turma em 2024.",
        "sql_ouro": "SELECT Turma, AVG(ano_ingresso) AS media_ingresso FROM dados WHERE Ano = 2024 GROUP BY Turma ORDER BY Turma",
        "esperado_cols": ["Turma", "media_ingresso"],
    },
    {
        "id": "i13",
        "categoria": "idade_data",
        "pergunta": "Amplitude de idade em anos em 2023 (máximo − mínimo).",
        "sql_ouro": """SELECT MAX(Ano - year(TRY_CAST(data_nasc AS DATE))) - MIN(Ano - year(TRY_CAST(data_nasc AS DATE))) AS amplitude_idade
FROM dados WHERE Ano = 2023 AND TRY_CAST(data_nasc AS DATE) IS NOT NULL""",
        "esperado_cols": ["amplitude_idade"],
    },
    {
        "id": "i14",
        "categoria": "idade_data",
        "pergunta": "Média de IPV por idade em anos arredondada em 2022.",
        "sql_ouro": """SELECT ROUND(Ano - year(TRY_CAST(data_nasc AS DATE))) AS idade_arred,
       AVG(IPV) AS media_ipv
FROM dados WHERE Ano = 2022 AND TRY_CAST(data_nasc AS DATE) IS NOT NULL
GROUP BY 1
ORDER BY 1""",
        "esperado_cols": ["idade_arred", "media_ipv"],
    },
    # --- multi_indicador (10) ---
    {
        "id": "x1",
        "categoria": "multi_indicador",
        "pergunta": "Em 2022, por Fase, qual a média de INDE e IDA e o gap INDE−IDA?",
        "sql_ouro": """WITH agg AS (
  SELECT Fase, AVG(INDE) AS media_inde, AVG(IDA) AS media_ida
  FROM dados WHERE Ano = 2022 GROUP BY Fase
)
SELECT Fase, media_inde, media_ida, (media_inde - media_ida) AS gap_inde_ida FROM agg ORDER BY Fase""",
        "esperado_cols": ["Fase", "media_inde", "media_ida", "gap_inde_ida"],
    },
    {
        "id": "x2",
        "categoria": "multi_indicador",
        "pergunta": "Ano a ano, média de IPV e média de IEG.",
        "sql_ouro": "SELECT Ano, AVG(IPV) AS media_ipv, AVG(IEG) AS media_ieg FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "media_ipv", "media_ieg"],
    },
    {
        "id": "x3",
        "categoria": "multi_indicador",
        "pergunta": "Em 2024, correlação simples: média de INDE e IDA por Turma.",
        "sql_ouro": "SELECT Turma, AVG(INDE) AS media_inde, AVG(IDA) AS media_ida FROM dados WHERE Ano = 2024 GROUP BY Turma ORDER BY Turma",
        "esperado_cols": ["Turma", "media_inde", "media_ida"],
    },
    {
        "id": "x4",
        "categoria": "multi_indicador",
        "pergunta": "Média de IAN e IEG na Fase 8 em 2023.",
        "sql_ouro": "SELECT AVG(IAN) AS media_ian, AVG(IEG) AS media_ieg FROM dados WHERE Fase = 8 AND Ano = 2023",
        "esperado_cols": ["media_ian", "media_ieg"],
    },
    {
        "id": "x5",
        "categoria": "multi_indicador",
        "pergunta": "Para cada Pedra, média de INDE e contagem em 2022.",
        "sql_ouro": "SELECT Pedra, AVG(INDE) AS media_inde, COUNT(*) AS n FROM dados WHERE Ano = 2022 GROUP BY Pedra ORDER BY Pedra",
        "esperado_cols": ["Pedra", "media_inde", "n"],
    },
    {
        "id": "x6",
        "categoria": "multi_indicador",
        "pergunta": "Em 2023, gap INDE−IDA médio por Turma.",
        "sql_ouro": """WITH t AS (
  SELECT Turma, AVG(INDE) AS mi, AVG(IDA) AS ma FROM dados WHERE Ano = 2023 GROUP BY Turma
)
SELECT Turma, mi, ma, (mi - ma) AS gap_inde_ida FROM t ORDER BY Turma""",
        "esperado_cols": ["Turma", "mi", "ma", "gap_inde_ida"],
    },
    {
        "id": "x7",
        "categoria": "multi_indicador",
        "pergunta": "Ano a ano, diferença entre média de IPV e média de IEG.",
        "sql_ouro": "SELECT Ano, AVG(IPV) - AVG(IEG) AS diff_ipv_ieg FROM dados GROUP BY Ano ORDER BY Ano",
        "esperado_cols": ["Ano", "diff_ipv_ieg"],
    },
    {
        "id": "x8",
        "categoria": "multi_indicador",
        "pergunta": "Em 2021, médias de INDE, IDA e IAN por Fase.",
        "sql_ouro": "SELECT Fase, AVG(INDE) AS media_inde, AVG(IDA) AS media_ida, AVG(IAN) AS media_ian FROM dados WHERE Ano = 2021 GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "media_inde", "media_ida", "media_ian"],
    },
    {
        "id": "x9",
        "categoria": "multi_indicador",
        "pergunta": "Média de INDE e contagem por Pedra em 2024.",
        "sql_ouro": "SELECT Pedra, AVG(INDE) AS media_inde, COUNT(*) AS n FROM dados WHERE Ano = 2024 GROUP BY Pedra ORDER BY Pedra",
        "esperado_cols": ["Pedra", "media_inde", "n"],
    },
    {
        "id": "x10",
        "categoria": "multi_indicador",
        "pergunta": "Em 2022, média de IAN e de IPV por Fase.",
        "sql_ouro": "SELECT Fase, AVG(IAN) AS media_ian, AVG(IPV) AS media_ipv FROM dados WHERE Ano = 2022 GROUP BY Fase ORDER BY Fase",
        "esperado_cols": ["Fase", "media_ian", "media_ipv"],
    },
    # --- ranking_topn (8) ---
    {
        "id": "k1",
        "categoria": "ranking_topn",
        "pergunta": "Top 5 turmas com maior média de IDA em 2022.",
        "sql_ouro": "SELECT Turma, AVG(IDA) AS media_ida FROM dados WHERE Ano = 2022 GROUP BY Turma ORDER BY media_ida DESC LIMIT 5",
        "esperado_cols": ["Turma", "media_ida"],
    },
    {
        "id": "k2",
        "categoria": "ranking_topn",
        "pergunta": "Fase com menor média de INDE em 2021 (apenas uma linha).",
        "sql_ouro": "SELECT Fase, AVG(INDE) AS media_inde FROM dados WHERE Ano = 2021 GROUP BY Fase ORDER BY media_inde ASC LIMIT 1",
        "esperado_cols": ["Fase", "media_inde"],
    },
    {
        "id": "k3",
        "categoria": "ranking_topn",
        "pergunta": "Top 3 níveis de Pedra com maior média de risco em 2024.",
        "sql_ouro": "SELECT Pedra, AVG(risco) AS media_risco FROM dados WHERE Ano = 2024 GROUP BY Pedra ORDER BY media_risco DESC LIMIT 3",
        "esperado_cols": ["Pedra", "media_risco"],
    },
    {
        "id": "k4",
        "categoria": "ranking_topn",
        "pergunta": "RA com menor média de IDA na Fase 6 (top 1).",
        "sql_ouro": "SELECT RA, AVG(IDA) AS media_ida FROM dados WHERE Fase = 6 GROUP BY RA ORDER BY media_ida ASC LIMIT 1",
        "esperado_cols": ["RA", "media_ida"],
    },
    {
        "id": "k5",
        "categoria": "ranking_topn",
        "pergunta": "Top 2 Fases com maior média de IAN em 2024.",
        "sql_ouro": "SELECT Fase, AVG(IAN) AS media_ian FROM dados WHERE Ano = 2024 GROUP BY Fase ORDER BY media_ian DESC LIMIT 2",
        "esperado_cols": ["Fase", "media_ian"],
    },
    {
        "id": "k6",
        "categoria": "ranking_topn",
        "pergunta": "As 3 turmas com menor média de INDE em 2021.",
        "sql_ouro": "SELECT Turma, AVG(INDE) AS media_inde FROM dados WHERE Ano = 2021 GROUP BY Turma ORDER BY media_inde ASC LIMIT 3",
        "esperado_cols": ["Turma", "media_inde"],
    },
    {
        "id": "k7",
        "categoria": "ranking_topn",
        "pergunta": "Top 10 RA por média de IDA em 2022.",
        "sql_ouro": "SELECT RA, AVG(IDA) AS media_ida FROM dados WHERE Ano = 2022 GROUP BY RA ORDER BY media_ida DESC LIMIT 10",
        "esperado_cols": ["RA", "media_ida"],
    },
    {
        "id": "k8",
        "categoria": "ranking_topn",
        "pergunta": "Pedra com menor média de IDA em 2023 (uma linha).",
        "sql_ouro": "SELECT Pedra, AVG(IDA) AS media_ida FROM dados WHERE Ano = 2023 GROUP BY Pedra ORDER BY media_ida ASC LIMIT 1",
        "esperado_cols": ["Pedra", "media_ida"],
    },
    # --- subset_complexo (7) ---
    {
        "id": "s1",
        "categoria": "subset_complexo",
        "pergunta": "Compare média IDA por turma com média da fase no mesmo ano (2021).",
        "sql_ouro": """
WITH por_turma AS (
  SELECT Turma, Fase, AVG(IDA) AS media_ida_turma
  FROM dados WHERE Ano = 2021 AND Fase IN (6, 7, 8)
  GROUP BY Turma, Fase
),
por_fase AS (
  SELECT Fase, AVG(IDA) AS media_ida_fase
  FROM dados WHERE Ano = 2021 AND Fase IN (6, 7, 8)
  GROUP BY Fase
)
SELECT pt.Turma, pt.Fase, pt.media_ida_turma, pf.media_ida_fase,
       pt.media_ida_turma - pf.media_ida_fase AS diff_vs_fase
FROM por_turma pt
JOIN por_fase pf ON pt.Fase = pf.Fase
ORDER BY pt.media_ida_turma ASC
LIMIT 5
""".strip(),
        "esperado_cols": ["Turma", "Fase", "media_ida_turma", "media_ida_fase", "diff_vs_fase"],
    },
    {
        "id": "s2",
        "categoria": "subset_complexo",
        "pergunta": "Contagem por Pedra em 2022 e 2023 lado a lado (pivot simples).",
        "sql_ouro": """
SELECT Pedra,
       SUM(CASE WHEN Ano = 2022 THEN 1 ELSE 0 END) AS n_2022,
       SUM(CASE WHEN Ano = 2023 THEN 1 ELSE 0 END) AS n_2023
FROM dados
GROUP BY Pedra
ORDER BY Pedra
""".strip(),
        "esperado_cols": ["Pedra", "n_2022", "n_2023"],
    },
    {
        "id": "s3",
        "categoria": "subset_complexo",
        "pergunta": "Para RA-MUD, listar Ano, Pedra e INDE (histórico).",
        "sql_ouro": "SELECT Ano, Pedra, INDE FROM dados WHERE RA = 'RA-MUD' ORDER BY Ano",
        "esperado_cols": ["Ano", "Pedra", "INDE"],
    },
    {
        "id": "s4",
        "categoria": "subset_complexo",
        "pergunta": "Média de INDE em 2024 apenas para quem tem risco abaixo de 0.5.",
        "sql_ouro": "SELECT AVG(INDE) AS media_inde FROM dados WHERE Ano = 2024 AND risco < 0.5",
        "esperado_cols": ["media_inde"],
    },
    {
        "id": "s5",
        "categoria": "subset_complexo",
        "pergunta": "Em 2022, média de IDA por turma vs média global da rede (pública/particular).",
        "sql_ouro": """
WITH base AS (
  SELECT Turma,
         CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
         AVG(IDA) AS media_ida_turma
  FROM dados WHERE Ano = 2022
  GROUP BY Turma, 2
),
media_rede AS (
  SELECT CASE WHEN escola_publica = true OR escola_publica = 1 THEN 'Pública' ELSE 'Particular' END AS tipo_rede,
         AVG(IDA) AS media_ida_rede
  FROM dados WHERE Ano = 2022
  GROUP BY 1
)
SELECT b.Turma, b.tipo_rede, b.media_ida_turma, g.media_ida_rede,
       b.media_ida_turma - g.media_ida_rede AS diff_vs_rede
FROM base b
JOIN media_rede g ON b.tipo_rede = g.tipo_rede
ORDER BY b.Turma
LIMIT 12
""".strip(),
        "esperado_cols": ["Turma", "tipo_rede", "media_ida_turma", "media_ida_rede", "diff_vs_rede"],
    },
    {
        "id": "s6",
        "categoria": "subset_complexo",
        "pergunta": "Alunos com média de INDE acima da média global em 2023 (contagem).",
        "sql_ouro": """
WITH g AS (SELECT AVG(INDE) AS m FROM dados WHERE Ano = 2023),
r AS (SELECT RA, AVG(INDE) AS mi FROM dados WHERE Ano = 2023 GROUP BY RA)
SELECT COUNT(*) AS n_acima FROM r, g WHERE r.mi > g.m
""".strip(),
        "esperado_cols": ["n_acima"],
    },
    {
        "id": "s7",
        "categoria": "subset_complexo",
        "pergunta": "Variação ano a ano da média de IPV (diferença face ao ano anterior).",
        "sql_ouro": """
WITH a AS (SELECT Ano, AVG(IPV) AS m FROM dados GROUP BY Ano)
SELECT Ano, m,
       m - LAG(m) OVER (ORDER BY Ano) AS delta_ipv
FROM a
ORDER BY Ano
""".strip(),
        "esperado_cols": ["Ano", "m", "delta_ipv"],
    },
]
