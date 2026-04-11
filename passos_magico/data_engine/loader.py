"""Carga do Parquet — funções puras; cache Streamlit fica em app.cached_data."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PARQUET = DATA_DIR / "dados.parquet"


def get_parquet_path() -> Path:
    env = os.environ.get("PASSOS_PARQUET")
    if env:
        return Path(env)
    return DEFAULT_PARQUET


def load_dados_df(parquet_path: Path | None = None) -> pd.DataFrame:
    path = parquet_path or get_parquet_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {path}. Execute scripts/etl.py primeiro."
        )
    return pd.read_parquet(path)
