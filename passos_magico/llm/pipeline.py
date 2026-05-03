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
    INSIGHT_SYSTEM_INSTITUTIONAL_ONLY,
    SQL_EXECUTION_FIX_APPEND,
    SQL_FAILURE_GUIDE_SYSTEM,
    SQL_GENERATION_RETRY_SUFFIX,
    SQL_GENERATION_SYSTEM,
    SUGGESTIONS_SYSTEM,
    THEO_SYSTEM_BASE,
    build_institutional_insight_user,
    build_insight_user,
    build_sql_execution_fix_user_message,
    build_sql_failure_user_message,
    build_sql_user_message,
    build_suggestions_user,
)
from passos_magico.llm.insight_mode import infer_insight_response_mode
from passos_magico.llm.kpi_narration import kpi_narration_block
from passos_magico.llm.sql_parse import (
    extract_json_suggestions,
    extract_sql_block,
    sql_passes_quick_validation,
)


@dataclass
class ChatTurnResult:
    sql: str | None
    df: pd.DataFrame | None
    sql_error: str | None
    figure: object | None
    chart_kind: str
    insight_text: str
    suggestions: list[str]
    recovery_markdown: str | None = None  # texto útil quando SQL não foi gerado/extraído


def _df_preview(df: pd.DataFrame, max_rows: int = 40) -> str:
    return df.head(max_rows).to_string(index=False)


def sql_and_chart_step(
    user_question: str,
    dictionary_block: str,
    ollama_ok: bool,
    sql_executor: Callable[[str], pd.DataFrame] | None = None,
    sql_context_df: pd.DataFrame | None = None,
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
    dados_columns: list[str] | None = None
    if sql_context_df is not None and not sql_context_df.empty:
        dados_columns = [str(c) for c in sql_context_df.columns]

    sys_sql = THEO_SYSTEM_BASE + "\n" + SQL_GENERATION_SYSTEM
    sys_sql_fix = sys_sql + "\n" + SQL_EXECUTION_FIX_APPEND
    user_sql = build_sql_user_message(user_question, dictionary_block, dados_columns=dados_columns)
    raw_sql = invoke_string(sys_sql, user_sql, temperature=0.05)
    sql = extract_sql_block(raw_sql)
    if not sql:
        raw_retry = invoke_string(
            sys_sql,
            user_sql + "\n\n" + SQL_GENERATION_RETRY_SUFFIX,
            temperature=0.02,
        )
        sql = extract_sql_block(raw_retry)
    if not sql:
        recovery: str | None = None
        try:
            recovery = invoke_string(
                SQL_FAILURE_GUIDE_SYSTEM,
                build_sql_failure_user_message(user_question),
                temperature=0.2,
            )
        except Exception:
            recovery = None
        return ChatTurnResult(
            sql=None,
            df=None,
            sql_error="Não foi possível extrair SQL da resposta do modelo.",
            figure=None,
            chart_kind="erro",
            insight_text="",
            suggestions=[],
            recovery_markdown=recovery,
        )

    _MAX_FIX_TRIES = 3
    last_exec_error: str | None = None
    for fix_round in range(_MAX_FIX_TRIES):
        if not sql_passes_quick_validation(sql):
            last_exec_error = (
                "SQL extraído parece incompleto (parênteses desbalanceados, falta FROM dados, "
                "ou estrutura de CTE quebrada)."
            )
            if fix_round < _MAX_FIX_TRIES - 1:
                raw_fix = invoke_string(
                    sys_sql_fix,
                    build_sql_execution_fix_user_message(
                        user_question,
                        dictionary_block,
                        sql,
                        last_exec_error,
                        dados_columns=dados_columns,
                    ),
                    temperature=0.02,
                )
                new_sql = extract_sql_block(raw_fix)
                sql = new_sql if new_sql else sql
                continue
            return ChatTurnResult(
                sql=sql,
                df=None,
                sql_error=last_exec_error,
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
            last_exec_error = str(e)
            if fix_round < _MAX_FIX_TRIES - 1:
                raw_fix = invoke_string(
                    sys_sql_fix,
                    build_sql_execution_fix_user_message(
                        user_question,
                        dictionary_block,
                        sql,
                        last_exec_error,
                        dados_columns=dados_columns,
                    ),
                    temperature=0.02,
                )
                new_sql = extract_sql_block(raw_fix)
                sql = new_sql if new_sql else sql
                continue
            return ChatTurnResult(
                sql=sql,
                df=None,
                sql_error=last_exec_error,
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

    return ChatTurnResult(
        sql=sql,
        df=None,
        sql_error=last_exec_error or "Falha ao gerar SQL executável.",
        figure=None,
        chart_kind="erro",
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
    kpi_block = kpi_narration_block(base.df)
    insight_mode = infer_insight_response_mode(base.df, base.chart_kind)
    insight_user = build_insight_user(
        user_question,
        preview,
        base.chart_kind,
        theo_context_block=dictionary_block,
        kpi_automatico=kpi_block,
        insight_mode=insight_mode,
    )
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
    theo_context_block: str,
    df: pd.DataFrame,
    chart_kind: str,
):
    """Gerador para streaming da explicação (segunda chamada ao modelo)."""
    from passos_magico.llm.ollama_client import stream_tokens

    preview = _df_preview(df)
    kpi_block = kpi_narration_block(df)
    insight_mode = infer_insight_response_mode(df, chart_kind)
    insight_user = build_insight_user(
        user_question,
        preview,
        chart_kind,
        theo_context_block=theo_context_block,
        kpi_automatico=kpi_block,
        insight_mode=insight_mode,
    )
    yield from stream_tokens(
        THEO_SYSTEM_BASE + INSIGHT_SYSTEM,
        insight_user,
        temperature=0.15,
    )


def stream_institutional_insight_text(
    user_question: str,
    theo_context_block: str,
):
    """Resposta só com texto do resumo anual — sem SQL na tabela `dados`."""
    from passos_magico.llm.ollama_client import stream_tokens

    user_msg = build_institutional_insight_user(user_question, theo_context_block)
    yield from stream_tokens(
        THEO_SYSTEM_BASE + INSIGHT_SYSTEM_INSTITUTIONAL_ONLY,
        user_msg,
        temperature=0.15,
    )


def invoke_institutional_insight(user_question: str, theo_context_block: str) -> str:
    """Versão não-streaming (fallback se o stream vier vazio ou genérico)."""
    user_msg = build_institutional_insight_user(user_question, theo_context_block)
    return invoke_string(
        THEO_SYSTEM_BASE + INSIGHT_SYSTEM_INSTITUTIONAL_ONLY,
        user_msg,
        temperature=0.15,
    )
