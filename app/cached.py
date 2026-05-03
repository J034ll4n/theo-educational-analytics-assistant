"""Wrappers com cache Streamlit para dados, SQL e modelo."""

from __future__ import annotations

import streamlit as st

from passos_magico.data_engine.loader import load_dados_df
from passos_magico.data_engine.query import run_sql
from passos_magico.ml.inference import ensure_risco_column, load_model_bundle


@st.cache_data(show_spinner=False)
def cached_load_dados():
    return load_dados_df()


def make_chat_sql_runner(df, bundle):
    """DataFrame DuckDB com coluna `risco` quando o modelo permite, e função `runner(sql)` (sem cache por SQL)."""
    df_sql = ensure_risco_column(df.copy(), bundle)

    def runner(sql: str):
        return run_sql(sql, df=df_sql, bundle=None)

    return df_sql, runner


@st.cache_resource
def cached_model_bundle():
    return load_model_bundle()
