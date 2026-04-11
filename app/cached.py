"""Wrappers com cache Streamlit para dados, SQL e modelo."""

from __future__ import annotations

import streamlit as st

from passos_magico.data_engine.loader import load_dados_df
from passos_magico.data_engine.query import run_sql
from passos_magico.ml.inference import load_model_bundle


@st.cache_data(show_spinner=False)
def cached_load_dados():
    return load_dados_df()


@st.cache_data(show_spinner=False)
def cached_sql_result(sql: str):
    """Cache de consultas idênticas (mesmo SQL)."""
    return run_sql(sql)


@st.cache_resource
def cached_model_bundle():
    return load_model_bundle()
