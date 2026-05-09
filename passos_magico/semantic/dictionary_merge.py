"""Fusão do dicionário editável com o esquema real do Parquet (evita desalinhamento Theo × dados)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Descrições canónicas (Passos Mágicos / PEDE) — chave = nome da coluna **no DataFrame após normalize_tabular_dataframe**.
STOCK_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "RA": (
        "Registro Acadêmico (identificador único do aluno na base). Use sempre este campo para filtrar ou agrupar por pessoa, "
        "não invente outro ID."
    ),
    "Nome": "Nome do aluno (pode estar anonimizado conforme export).",
    "Fase": (
        "Fase do programa Passos Mágicos (inteiro 1–8). Representa o nível/ciclo em que o aluno está acompanhado; "
        "não confundir com série escolar genérica fora do programa."
    ),
    "Turma": (
        "Letra da turma dentro da fase (tipicamente A–E). Valores fora de A–E podem existir no bruto; na modelagem usa-se também Turma_ord."
    ),
    "Ano": (
        "Ano de referência da linha (calendário da ficha/indicadores), inteiro (ex.: 2020–2024). "
        "É a coluna correta para filtrar por ano letivo — **não** existe coluna chamada `year`."
    ),
    "INDE": (
        "Índice de Desenvolvimento Educacional (escala típica 0–10). Síntese global do desempenho no modelo pedagógico da Passos Mágicos."
    ),
    "IDA": "Indicador de Aprendizagem — eixo de desempenho académico / domínio de conteúdos.",
    "IAN": (
        "Indicador de Adequação ao Nível — alinhamento entre idade, fase e expectativa de aprendizagem "
        "(não é simplesmente «nota baixa»; cruza com fase e histórico)."
    ),
    "IEG": "Indicador de Engajamento — participação, entrega de atividades e envolvimento com as propostas.",
    "IPV": "Indicador de Ponto de Virada — mudanças relevantes de percurso ou salto qualitativo observado.",
    "IAA": "Indicador de Autoavaliação (quando existir no export); eixo de reflexão do aluno sobre o próprio percurso.",
    "IPS": "Indicador psicossocial / socioemocional (quando existir no export).",
    "IPP": "Indicador de percepção de progresso ou projeção (quando existir no export).",
    "MAT": "Desempenho ou nota associada à componente de Matemática (quando existir).",
    "POR": "Desempenho ou nota associada à componente de Português (quando existir).",
    "ING": "Desempenho ou nota associada à componente de Inglês (quando existir).",
    "Pedra": (
        "Classificação de trajetória por mérito (Quartzo, Ágata, Ametista, Topázio). "
        "É categórica — comparar distribuições com GROUP BY ou CASE, não somar como número contínuo sem contexto."
    ),
    "Genero": "Género declarado/registado (texto ou código conforme export).",
    "risco": (
        "Probabilidade entre 0 e 1 de **alto risco escolar** (defasagem/evasão) produzida pelo modelo XGBoost em produção. "
        "Na UI usa-se limiar operacional ~0,46; em SQL use a coluna **risco** exatamente assim (minúsculas)."
    ),
    "data_nasc": (
        "Data de nascimento em texto (muitas vezes ISO `YYYY-MM-DD`). Para idade ou ano de nascimento em SQL use "
        "`TRY_CAST(data_nasc AS DATE)` antes de `year()` ou `date_part` — não aplique funções de data direto em VARCHAR."
    ),
    "defasado": "Flag booleana (ou 0/1) indicando defasagem escolar segundo regra do export — interpretar com o dicionário institucional.",
    "escola_publica": "Indica se a escola de origem é pública (true/1) ou não (false/0), conforme o export.",
    "ano_ingresso": "Ano de entrada do aluno no programa ou na instituição (inteiro), quando existir.",
    "Instituicao_de_ensino": "Identificador ou nome da instituição de ensino de origem (quando existir no Parquet).",
    # --- Colunas derivadas / agregadas frequentes no notebook ou Parquet enriquecido (snake_case) ---
    "idade": "Idade em anos derivada (Ano da ficha vs. nascimento) ou coluna já calculada no pipeline de risco — numérica.",
    "iaa": "Alias minúsculo de IAA quando coexistir no Parquet mesclado; se ambas existirem após normalização, preferir a coluna canónica IAA.",
    "ida": "Alias minúsculo de IDA (id.).",
    "ian": "Alias minúsculo de IAN (id.).",
    "ieg": "Alias minúsculo de IEG (id.).",
    "ipv": "Alias minúsculo de IPV (id.).",
    "inde": "Alias minúsculo de INDE (id.).",
    "ips": "Alias minúsculo de IPS (id.).",
    "ipp": "Alias minúsculo de IPP (id.).",
    "mat": "Alias minúsculo de MAT (id.).",
    "por": "Alias minúsculo de POR (id.).",
    "ing": "Alias minúsculo de ING (id.).",
    "fase": "Alias minúsculo de Fase (id.); preferir **Fase** se existir no mesmo ficheiro.",
    "turma": "Alias minúsculo de Turma (id.).",
    "pedra": "Alias minúsculo de Pedra (id.).",
    "genero": "Alias minúsculo de Genero (id.).",
    "instituicao_de_ensino": "Chave da instituição (texto) usada no modelo de risco — categórica.",
    "defas": "Indicador numérico de defasagem associado ao pipeline de risco (quando existir).",
    "delta_inde": "Variação do INDE em relação ao período/registo anterior (agregado por aluno quando calculado).",
    "delta_ian": "Variação do IAN vs. período anterior (id.).",
    "std_inde": "Desvio padrão do INDE ao longo do histórico do aluno (agregado).",
    "media_inde": "Média do INDE no histórico do aluno (agregado).",
    "tendencia_inde": "Tendência (inclinação) da série de INDE do aluno (agregado).",
    "queda_acumulada_inde": "Soma acumulada de quedas de INDE entre períodos (agregado).",
    "range_inde": "Amplitude melhor–pior INDE no histórico (agregado).",
    "pior_inde": "Valor mínimo de INDE observado no histórico (agregado).",
    "melhor_inde": "Valor máximo de INDE observado no histórico (agregado).",
    "media_turma_inde": "Média de INDE dos colegas do mesmo contexto (turma/instituição/fase conforme cálculo) — benchmark.",
    "distancia_media_turma": "Diferença entre INDE do aluno e média do grupo de referência (agregado).",
    "choque_realidade": "Indicador composto IAA vs. IDA (esforço percebido vs. resultado) quando existir.",
    "esforco_sem_resultado": "Indicador composto IEG vs. IDA quando existir.",
    "mudanca_pedra": "Flag ou contagem de mudança de Pedra entre anos (agregado).",
    "Turma_ord": "Ordinal 1–5 da turma (A=1…E=5) usado no modelo de risco — derivado de Turma.",
    "Pedra_ord": "Ordinal da Pedra usado no modelo de risco — derivado de Pedra.",
    "Idade": "Idade em anos (capitalizado) se existir como coluna explícita no export.",
}


def _fallback_description(col: str, dtype: object) -> str:
    dt = str(dtype)
    return (
        f"Coluna «{col}» no Parquet carregado (tipo {dt}). "
        "Use o nome exatamente como listado; não substitua por sinónimos em inglês nem por campos de relatórios PDF/Gamma."
    )


def merge_dictionary_with_dataframe(df: pd.DataFrame, rows: list[dict]) -> list[dict]:
    """
    Ordena e completa entradas do dicionário conforme as colunas **reais** de `df` (pós-ETL / normalize).
    Descrições editadas pelo utilizador em `dicionario.json` têm prioridade sobre o stock.
    """
    if df is None or df.empty:
        return list(rows)

    by_col: dict[str, dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        c = r.get("coluna")
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        c = str(c).strip()
        if c:
            by_col[c] = dict(r)

    out: list[dict] = []
    for col in df.columns:
        col = str(col)
        prev = by_col.get(col, {})
        user_desc = (prev.get("descricao") or "").strip()
        if user_desc:
            desc = user_desc
        else:
            desc = STOCK_COLUMN_DESCRIPTIONS.get(col, _fallback_description(col, df[col].dtype))
        entry: dict[str, str | object] = {"coluna": col, "descricao": desc}
        tipo_prev = prev.get("tipo")
        if tipo_prev is not None and str(tipo_prev).strip():
            entry["tipo"] = str(tipo_prev).strip()
        else:
            entry["tipo"] = str(df[col].dtype)
        out.append(entry)
    return out
