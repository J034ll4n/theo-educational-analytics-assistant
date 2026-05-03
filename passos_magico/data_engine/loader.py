"""Carga do Parquet — funções puras; cache Streamlit fica em app.cached_data."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_ML_PARQUET = PROJECT_ROOT / "notebooks" / "Ml" / "relatorio_final.parquet"
DEFAULT_PARQUET = DATA_DIR / "dados.parquet"

TRAINING_MISSING_SENTINELS: tuple[float | int, ...] = (-999, -999.0)


def get_parquet_path() -> Path:
    env = os.environ.get("PASSOS_PARQUET")
    if env:
        return Path(env)
    if NOTEBOOK_ML_PARQUET.exists():
        return NOTEBOOK_ML_PARQUET
    return DEFAULT_PARQUET


def replace_training_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Converte -999 (sentinela de ausente no treino) em NaN."""
    out = df.replace(dict.fromkeys(TRAINING_MISSING_SENTINELS, np.nan))
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].replace({"-999": np.nan})
    return out


def _fase_cell_to_float(x) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = re.search(r"(\d+)", str(x))
    return float(m.group(1)) if m else 0.0


# Par (coluna canónica UI/SQL, coluna snake_case redundante no Parquet mesclado).
ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("RA", "ra"),
    ("Nome", "nome"),
    ("Ano", "ano_referencia"),
    ("Turma", "turma"),
    ("Genero", "genero"),
    ("Pedra", "pedra"),
    ("INDE", "inde"),
    ("IDA", "ida"),
    ("IAN", "ian"),
    ("IEG", "ieg"),
    ("IPV", "ipv"),
    ("IAA", "iaa"),
    ("IPS", "ips"),
    ("IPP", "ipp"),
    ("MAT", "mat"),
    ("POR", "por"),
    ("ING", "ing"),
)


def _series_pair_compatible(hi: pd.Series, lo: pd.Series) -> bool:
    """True se os valores forem equivalentes para efeito de deduplicação."""
    if hi.shape != lo.shape:
        return False
    h_num = pd.to_numeric(hi, errors="coerce")
    l_num = pd.to_numeric(lo, errors="coerce")
    if h_num.notna().all() and l_num.notna().all():
        return bool(
            np.allclose(
                h_num.to_numpy(dtype=float),
                l_num.to_numpy(dtype=float),
                equal_nan=True,
                rtol=0.0,
                atol=1e-5,
            )
        )
    hs = hi.astype("string").fillna("<NA>")
    ls = lo.astype("string").fillna("<NA>")
    return bool((hs == ls).all())


def drop_redundant_snake_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas snake_case quando a coluna canónica já existe (evita duplicar esquema no DuckDB)."""
    out = df.copy()
    for hi, lo in ALIAS_PAIRS:
        if hi not in out.columns or lo not in out.columns:
            continue
        if not _series_pair_compatible(out[hi], out[lo]):
            warnings.warn(
                f"Colunas '{hi}' e '{lo}' coexistem com valores divergentes; mantém-se '{hi}' e remove-se '{lo}'.",
                stacklevel=2,
            )
        out = out.drop(columns=[lo])

    if "Fase" in out.columns and "fase" in out.columns:
        fase_from_lo = out["fase"].map(_fase_cell_to_float)
        mask_fill = out["Fase"].isna() & fase_from_lo.notna() & (fase_from_lo != 0.0)
        if mask_fill.any():
            out = out.copy()
            out.loc[mask_fill, "Fase"] = fase_from_lo.loc[mask_fill]
        f_hi = pd.to_numeric(out["Fase"], errors="coerce")
        if not np.allclose(
            f_hi.fillna(0).to_numpy(dtype=float),
            fase_from_lo.fillna(0).to_numpy(dtype=float),
            equal_nan=True,
            rtol=0.0,
            atol=1e-5,
        ):
            warnings.warn(
                "Colunas 'Fase' e 'fase' coexistem com valores divergentes; mantém-se 'Fase' e remove-se 'fase'.",
                stacklevel=2,
            )
        out = out.drop(columns=["fase"])
    return out


def ensure_streamlit_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Snake_case do relatório final → colunas esperadas pela UI (RA, Nome, INDE, …)."""
    out = df
    for hi, lo in ALIAS_PAIRS:
        if hi not in out.columns and lo in out.columns:
            out = out.copy()
            out[hi] = out[lo]
    if "Fase" not in out.columns and "fase" in out.columns:
        out = out.copy()
        out["Fase"] = out["fase"].map(_fase_cell_to_float)
    return out


def normalize_tabular_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Pós-processamento partilhado: sentinéis de treino, aliases UI, remoção de colunas snake duplicadas."""
    df = replace_training_sentinels(df)
    df = ensure_streamlit_aliases(df)
    df = drop_redundant_snake_columns(df)
    return df


def load_dados_df(parquet_path: Path | None = None) -> pd.DataFrame:
    path = parquet_path or get_parquet_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {path}. "
            "Defina PASSOS_PARQUET ou coloque notebooks/Ml/relatorio_final.parquet ou data/dados.parquet."
        )
    df = pd.read_parquet(path)
    return normalize_tabular_dataframe(df)
