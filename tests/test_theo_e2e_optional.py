"""Testes opcionais com Ollama real — `pytest -m ollama` (skip se Ollama off)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_community", reason="dependência opcional para LLM")

from passos_magico.llm.ollama_client import ollama_available
from passos_magico.llm.pipeline import sql_and_chart_step
from passos_magico.semantic.metadata import rows_to_prompt_block
from tests.fixtures.theo_question_catalog import build_catalog_demo_df, get_smoke_twenty_question_cases


pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def catalog_df():
    return build_catalog_demo_df()


@pytest.fixture(scope="module")
def dictionary_block():
    return rows_to_prompt_block(
        [
            {"coluna": "Ano", "descricao": "Ano de referência"},
            {"coluna": "IDA", "descricao": "Indicador"},
        ]
    )


def _run_sql(sql: str, df):
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        con.register("dados", df)
        return con.execute(sql).df()
    finally:
        con.close()


@pytest.mark.parametrize("case", get_smoke_twenty_question_cases(), ids=lambda c: c["id"])
def test_sql_and_chart_step_executes_with_ollama(case, catalog_df, dictionary_block):
    if not ollama_available():
        pytest.skip("Ollama não disponível")

    step = sql_and_chart_step(
        case["pergunta"],
        dictionary_block,
        True,
        sql_executor=lambda s: _run_sql(s, catalog_df.copy()),
        sql_context_df=catalog_df,
    )
    assert step.df is not None and not step.df.empty, (step.sql_error, step.sql)
