"""Orquestração do pipeline analítico (SQL → gráfico → insight → sugestões)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from passos_magico.data_engine.query import run_sql as default_run_sql
from passos_magico.data_engine.query import validate_select_only
from passos_magico.llm.charts import dataframe_to_figure
from passos_magico.llm.ollama_client import invoke_string
from passos_magico.llm.prompts import (
    INSIGHT_SYSTEM,
    SQL_GENERATION_SYSTEM,
    SUGGESTIONS_SYSTEM,
    THEO_SYSTEM_BASE,
    build_insight_user,
    build_sql_user_message,
    build_suggestions_user,
)
from passos_magico.llm.sql_parse import extract_json_suggestions, extract_sql_block


@dataclass
class ChatTurnResult:
    sql: str | None
    df: pd.DataFrame | None
    sql_error: str | None
    figure: object | None
    chart_kind: str
    insight_text: str
    suggestions: list[str]


def _df_preview(df: pd.DataFrame, max_rows: int = 40) -> str:
    return df.head(max_rows).to_string(index=False)


def sql_and_chart_step(
    user_question: str,
    dictionary_block: str,
    ollama_ok: bool,
    sql_executor: Callable[[str], pd.DataFrame] | None = None,
) -> ChatTurnResult:
    """Etapa 1–2: gera SQL, executa e monta gráfico (sem insight/sugestões)."""
    if not ollama_ok:
        return ChatTurnResult(
            sql=None,
            df=None,
            sql_error="Ollama não está disponível.",
            figure=None,
            chart_kind="erro",
            insight_text="",
            suggestions=[],
        )

    exec_sql = sql_executor or default_run_sql

    sys_sql = THEO_SYSTEM_BASE + "\n" + SQL_GENERATION_SYSTEM
    user_sql = build_sql_user_message(user_question, dictionary_block)
    raw_sql = invoke_string(sys_sql, user_sql, temperature=0.05)
    sql = extract_sql_block(raw_sql)
    if not sql:
        return ChatTurnResult(
            sql=None,
            df=None,
            sql_error="Não foi possível extrair SQL da resposta do modelo.",
            figure=None,
            chart_kind="erro",
            insight_text="",
            suggestions=[],
        )

    ok, err = validate_select_only(sql)
    if not ok:
        return ChatTurnResult(
            sql=sql,
            df=None,
            sql_error=err,
            figure=None,
            chart_kind="erro",
            insight_text="",
            suggestions=[],
        )

    try:
        df = exec_sql(sql)
    except Exception as e:
        return ChatTurnResult(
            sql=sql,
            df=None,
            sql_error=str(e),
            figure=None,
            chart_kind="erro",
            insight_text="",
            suggestions=[],
        )

    fig, kind = dataframe_to_figure(df)
    return ChatTurnResult(
        sql=sql,
        df=df,
        sql_error=None,
        figure=fig,
        chart_kind=kind,
        insight_text="",
        suggestions=[],
    )


def suggestions_step(user_question: str, insight_full: str) -> list[str]:
    """Etapa 4: três sugestões com base no insight já gerado."""
    sug_raw = invoke_string(
        SUGGESTIONS_SYSTEM,
        build_suggestions_user(user_question, insight_full[:800]),
        temperature=0.15,
    )
    return extract_json_suggestions(sug_raw)


def run_analytical_turn(
    user_question: str,
    dictionary_block: str,
    ollama_ok: bool,
) -> ChatTurnResult:
    """Fluxo completo síncrono (testes / fallback sem streaming)."""
    if not ollama_ok:
        return ChatTurnResult(
            sql=None,
            df=None,
            sql_error="Ollama não está disponível. Inicie o serviço local e verifique o modelo.",
            figure=None,
            chart_kind="erro",
            insight_text=(
                "**Theo:** Não foi possível conectar ao Ollama. "
                "Confira se o servidor está em execução e se o modelo configurado foi baixado (`ollama pull`)."
            ),
            suggestions=[
                "Após iniciar o Ollama, repita sua pergunta.",
                "Verificar média de IDA por ano.",
                "Comparar INDE entre fases.",
            ],
        )

    base = sql_and_chart_step(user_question, dictionary_block, True)
    if base.sql_error or base.df is None:
        if base.sql_error and "extrair SQL" in base.sql_error:
            return ChatTurnResult(
                sql=base.sql,
                df=None,
                sql_error=base.sql_error,
                figure=None,
                chart_kind="erro",
                insight_text=(
                    "**Theo:** Não consegui gerar uma consulta válida. "
                    "Reformule a pergunta ou seja mais específico sobre ano, fase ou indicador."
                ),
                suggestions=[
                    "Média de IDA por Fase no ano 2022",
                    "Contagem de alunos por Turma em 2021",
                    "Evolução do INDE por Ano",
                ],
            )
        if base.sql_error and base.sql:
            return ChatTurnResult(
                sql=base.sql,
                df=None,
                sql_error=base.sql_error,
                figure=None,
                chart_kind="erro",
                insight_text=f"**Theo:** A consulta gerada não passou na validação: {base.sql_error}",
                suggestions=[
                    "Repetir com filtros explícitos (Ano, Fase).",
                    "Pedir apenas uma agregação simples.",
                    "Solicitar as primeiras linhas da tabela com LIMIT.",
                ],
            )
        if base.sql_error:
            return ChatTurnResult(
                sql=base.sql,
                df=None,
                sql_error=base.sql_error,
                figure=None,
                chart_kind="erro",
                insight_text=(
                    f"**Theo:** Erro ao executar a consulta. Detalhes: `{base.sql_error}`."
                ),
                suggestions=[
                    "Listar média de IDA por Ano",
                    "Contar alunos por Pedra",
                    "Mostrar INDE médio por Turma no último ano",
                ],
            )

    assert base.df is not None
    preview = _df_preview(base.df)
    insight_user = build_insight_user(user_question, preview, base.chart_kind)
    insight = invoke_string(
        THEO_SYSTEM_BASE + INSIGHT_SYSTEM,
        insight_user,
        temperature=0.15,
    )
    sugs = suggestions_step(user_question, insight)
    return ChatTurnResult(
        sql=base.sql,
        df=base.df,
        sql_error=None,
        figure=base.figure,
        chart_kind=base.chart_kind,
        insight_text=insight,
        suggestions=sugs,
    )


def stream_insight_text(
    user_question: str,
    _dictionary_block: str,
    df: pd.DataFrame,
    chart_kind: str,
):
    """Gerador para streaming da explicação (segunda chamada ao modelo)."""
    from passos_magico.llm.ollama_client import stream_tokens

    preview = _df_preview(df)
    insight_user = build_insight_user(user_question, preview, chart_kind)
    yield from stream_tokens(
        THEO_SYSTEM_BASE + INSIGHT_SYSTEM,
        insight_user,
        temperature=0.15,
    )
