"""Testes do catálogo Theo + reescritas SQL (DuckDB)."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from passos_magico.data_engine.query import (
    apply_sql_rewrites,
    clear_recent_duckdb_errors,
    recent_duckdb_errors,
    rewrite_aggregate_in_group_by,
    run_sql,
    validate_select_only,
)
from passos_magico.llm.sql_parse import sql_passes_quick_validation
from tests.fixtures.theo_question_catalog import (
    CATEGORIA_PESO_RELATIVA,
    QUESTION_CATALOG,
    QuestionCase,
    amostra_perguntas_por_peso,
    build_catalog_demo_df,
)


@pytest.fixture(scope="module")
def catalog_df() -> pd.DataFrame:
    return build_catalog_demo_df()


@pytest.fixture
def fixture_string_data_nasc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RA": ["RA1", "RA2"],
            "Nome": ["A", "B"],
            "Fase": [8, 8],
            "Turma": ["A", "A"],
            "Ano": [2024, 2024],
            "INDE": [6.0, 6.0],
            "IDA": [5.0, 6.0],
            "IAN": [6.0, 6.0],
            "IEG": [6.0, 6.0],
            "IPV": [6.0, 6.0],
            "Pedra": ["Quartzo", "Quartzo"],
            "risco": [0.3, 0.5],
            "data_nasc": pd.Series(["2005-06-01", "2010-01-15"], dtype="string"),
            "escola_publica": [True, False],
            "ano_ingresso": [2020, 2021],
            "defasado": [False, False],
        }
    )


@pytest.mark.parametrize("case", QUESTION_CATALOG, ids=lambda c: c["id"])
def test_sql_ouro_passa_validacao(case: QuestionCase) -> None:
    ok, err = validate_select_only(case["sql_ouro"])
    assert ok, err
    assert sql_passes_quick_validation(case["sql_ouro"])


@pytest.mark.parametrize("case", QUESTION_CATALOG, ids=lambda c: c["id"])
def test_sql_ouro_executa_no_duckdb(case: QuestionCase, catalog_df: pd.DataFrame) -> None:
    df = run_sql(case["sql_ouro"], df=catalog_df.copy())
    for col in case["esperado_cols"]:
        assert col in df.columns, f"Falta coluna {col!r} em {list(df.columns)}"


@pytest.mark.parametrize("case", QUESTION_CATALOG, ids=lambda c: c["id"])
def test_sql_ouro_apos_reescrita(case: QuestionCase, catalog_df: pd.DataFrame) -> None:
    rewritten = apply_sql_rewrites(case["sql_ouro"], catalog_df)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("dados", catalog_df)
        if "limit" not in rewritten.lower():
            rewritten = f"{rewritten} LIMIT 5000"
        df = con.execute(rewritten).df()
    finally:
        con.close()
    for col in case["esperado_cols"]:
        assert col in df.columns


def test_regression_date_part_varchar(fixture_string_data_nasc: pd.DataFrame) -> None:
    sql = "SELECT AVG(date_part('year', data_nasc)) AS x FROM dados"
    df = run_sql(sql, df=fixture_string_data_nasc)
    assert "x" in df.columns
    assert df["x"].notna().all()


def test_regression_integer_minus_date(fixture_string_data_nasc: pd.DataFrame) -> None:
    sql = "SELECT AVG(2024 - TRY_CAST(data_nasc AS DATE)) AS x FROM dados"
    df = run_sql(sql, df=fixture_string_data_nasc)
    assert "x" in df.columns


def test_regression_aggregate_in_group_by(catalog_df: pd.DataFrame) -> None:
    sql = "SELECT AVG(IDA) AS m FROM dados GROUP BY AVG(IDA)"
    df = run_sql(sql, df=catalog_df.copy())
    assert "m" in df.columns
    assert len(df) >= 1


def test_group_by_parenthesized_expression_preserved() -> None:
    sql = "SELECT 1 AS a FROM dados GROUP BY (Ano + Fase) ORDER BY 1"
    df = pd.DataFrame({"Ano": [2024], "Fase": [8]})
    out = rewrite_aggregate_in_group_by(sql)
    assert "Ano + Fase" in out


def test_recent_duck_errors_recorded() -> None:
    clear_recent_duckdb_errors()
    bad = "SELECT nao_existe FROM dados"
    with pytest.raises(duckdb.Error):
        run_sql(bad, df=build_catalog_demo_df())
    errs = recent_duckdb_errors()
    assert len(errs) >= 1


def test_question_catalog_cem_entradas_e_pesos_normalizados() -> None:
    assert len(QUESTION_CATALOG) == 100
    assert abs(sum(CATEGORIA_PESO_RELATIVA.values()) - 1.0) < 1e-9


def test_amostra_perguntas_por_peso_reprodutivel() -> None:
    a = amostra_perguntas_por_peso(k=8, seed=42)
    b = amostra_perguntas_por_peso(k=8, seed=42)
    assert a == b
    assert len(a) == 8
    assert all(isinstance(p, str) and p for p in a)
