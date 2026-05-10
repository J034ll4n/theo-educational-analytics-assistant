"""Features e colunas do modelo `modelo_risco_aluno.pkl` (notebook ML_Passos_Magicos)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from passos_magico.ml.features import (
    latest_single_row_for_ra,
    pick_latest_year_row,
    single_row_for_ra_and_year,
)

RISK_NUM_FEATURES: list[str] = [
    "idade",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipp",
    "ipv",
    "ian",
    "mat",
    "por",
    "ing",
    "defas",
    "delta_inde",
    "delta_ian",
    "std_inde",
    "media_inde",
    "tendencia_inde",
    "queda_acumulada_inde",
    "range_inde",
    "distancia_media_turma",
    "choque_realidade",
    "esforco_sem_resultado",
    "mudanca_pedra",
    "tem_ingles",
]
RISK_CAT_FEATURES: list[str] = ["fase", "genero", "instituicao_de_ensino", "pedra"]
RISK_MODEL_COLUMNS: list[str] = RISK_NUM_FEATURES + RISK_CAT_FEATURES

_PEDRA_NUM = {"Quartzo": 1, "Ágata": 2, "Agata": 2, "Ametista": 3, "Topázio": 4, "Topazio": 4}
_REV_PEDRA = {1: "Quartzo", 2: "Ágata", 3: "Ametista", 4: "Topázio"}
_TURMA_ORD_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


def _coerce_fase_number(val: Any) -> float:
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, np.integer)):
        return float(val)
    if isinstance(val, (float, np.floating)) and not np.isnan(val):
        return float(val)
    m = re.search(r"(\d+)", str(val))
    return float(m.group(1)) if m else 0.0


def _fase_string_from_number(n: float) -> str:
    return f"Fase {int(round(float(n)))}"


def _ensure_model_keys(d: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas minúsculas usadas no treino (`ra`, `inde`, …)."""
    out = d
    pairs = [
        ("ra", "RA"),
        ("nome", "Nome"),
        ("ano_referencia", "Ano"),
        ("inde", "INDE"),
        ("ian", "IAN"),
        ("ida", "IDA"),
        ("ieg", "IEG"),
        ("ipv", "IPV"),
        ("iaa", "IAA"),
        ("ips", "IPS"),
        ("ips", "ipsm"),
        ("ips", "IPSM"),
        ("ipp", "IPP"),
        ("ipp", "ippm"),
        ("ipp", "IPPM"),
        ("mat", "MAT"),
        ("por", "POR"),
        ("ing", "ING"),
        ("cg", "CG"),
        ("cf", "CF"),
        ("ct", "CT"),
        ("fase", "Fase"),
        ("turma", "Turma"),
        ("genero", "Genero"),
        ("pedra", "Pedra"),
        ("instituicao_de_ensino", "Instituicao_de_ensino"),
        ("idade", "Idade"),
    ]
    for lo, hi in pairs:
        if lo not in out.columns and hi in out.columns:
            out = out.copy()
            if lo == "fase":
                fv = out[hi]
                if pd.api.types.is_numeric_dtype(fv):
                    out[lo] = fv.map(_fase_string_from_number)
                else:
                    out[lo] = fv.astype(str)
            else:
                out[lo] = out[hi]
    return out


def _fill_idade_if_missing(out: pd.DataFrame) -> pd.DataFrame:
    """Garante coluna numérica `idade` (treino do modelo); Parquet costuma trazer só data de nascimento."""
    d = out.copy()
    if "idade" in d.columns:
        d["idade"] = pd.to_numeric(d["idade"], errors="coerce")

    ref = None
    for c in ("ano_referencia", "Ano"):
        if c in d.columns:
            ref = pd.to_numeric(d[c], errors="coerce")
            break

    computed = pd.Series(np.nan, index=d.index, dtype=float)
    if ref is not None and "ano_nasc" in d.columns:
        ano_n = pd.to_numeric(d["ano_nasc"], errors="coerce")
        computed = ref.astype(float) - ano_n
    elif ref is not None:
        for c in ("data_nasc", "Data_nasc", "data_nascimento", "Data_Nascimento"):
            if c not in d.columns:
                continue
            birth_y = pd.to_datetime(d[c], errors="coerce").dt.year
            computed = ref.astype(float) - birth_y.astype(float)
            break

    if "idade" not in d.columns:
        d["idade"] = computed
    else:
        d["idade"] = d["idade"].fillna(computed)

    med = float(d["idade"].median()) if d["idade"].notna().any() else 14.0
    if not np.isfinite(med):
        med = 14.0
    d["idade"] = pd.to_numeric(d["idade"], errors="coerce").fillna(med)
    return d


def ensure_risk_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Replica engenharia do notebook para colunas ausentes no Parquet."""
    out = _fill_idade_if_missing(_ensure_model_keys(df.copy()))
    # Alinhado ao treino (`ML_Passos_Magicos.ipynb`): indicador binário; Parquet antigo pode não trazer a coluna.
    if "tem_ingles" not in out.columns:
        out["tem_ingles"] = 0.0
    else:
        out["tem_ingles"] = pd.to_numeric(out["tem_ingles"], errors="coerce").fillna(0.0)
    # Notebook (Transform): IPP ≈ média de CF e CT quando o indicador composto não veio no ficheiro.
    if "cf" in out.columns and "ct" in out.columns:
        cf_n = pd.to_numeric(out["cf"], errors="coerce")
        ct_n = pd.to_numeric(out["ct"], errors="coerce")
        proxy_ipp = (cf_n + ct_n) / 2.0
        if "ipp" in out.columns:
            ipp_n = pd.to_numeric(out["ipp"], errors="coerce")
            out["ipp"] = ipp_n.where(ipp_n.notna(), proxy_ipp)
        else:
            out["ipp"] = proxy_ipp
    if "ra" not in out.columns or "ano_referencia" not in out.columns:
        return out
    if all(c in out.columns for c in RISK_MODEL_COLUMNS):
        return out
    out = out.sort_values(["ra", "ano_referencia"]).copy()
    grp = out.groupby("ra", sort=False)
    if "delta_inde" not in out.columns and "inde" in out.columns:
        out["delta_inde"] = grp["inde"].diff().fillna(0)
    if "delta_ian" not in out.columns and "ian" in out.columns:
        out["delta_ian"] = grp["ian"].diff().fillna(0)
    if "std_inde" not in out.columns and "inde" in out.columns:
        out["std_inde"] = grp["inde"].transform(lambda x: x.expanding().std()).fillna(0)
    if "media_inde" not in out.columns and "inde" in out.columns:
        out["media_inde"] = grp["inde"].transform(lambda x: x.expanding().mean())
    if "tendencia_inde" not in out.columns and "inde" in out.columns:
        out["tendencia_inde"] = (
            grp["inde"].transform(lambda x: x.diff().rolling(3).mean()).fillna(0)
        )
    if "queda_acumulada_inde" not in out.columns and "inde" in out.columns:
        out["queda_acumulada_inde"] = grp["inde"].transform(lambda x: x.iloc[0] - x)
    if "pior_inde" not in out.columns and "inde" in out.columns:
        out["pior_inde"] = grp["inde"].transform("min")
    if "melhor_inde" not in out.columns and "inde" in out.columns:
        out["melhor_inde"] = grp["inde"].transform("max")
    if "range_inde" not in out.columns:
        if "melhor_inde" in out.columns and "pior_inde" in out.columns:
            out["range_inde"] = out["melhor_inde"] - out["pior_inde"]
    if "media_turma_inde" not in out.columns and "inde" in out.columns:
        gcols = [c for c in ["instituicao_de_ensino", "fase", "ano_referencia"] if c in out.columns]
        if len(gcols) >= 2:
            out["media_turma_inde"] = out.groupby(gcols)["inde"].transform("mean")
    if "distancia_media_turma" not in out.columns:
        if "media_turma_inde" in out.columns and "inde" in out.columns:
            out["distancia_media_turma"] = out["inde"] - out["media_turma_inde"]
        elif "inde" in out.columns:
            out["distancia_media_turma"] = 0.0
    if "choque_realidade" not in out.columns and "iaa" in out.columns and "ida" in out.columns:
        out["choque_realidade"] = out["iaa"] - out["ida"]
    if "esforco_sem_resultado" not in out.columns and "ieg" in out.columns and "ida" in out.columns:
        out["esforco_sem_resultado"] = out["ieg"] / (out["ida"] + 0.1)
    if "mudanca_pedra" not in out.columns and "pedra" in out.columns:
        out["pedra_num"] = out["pedra"].map(_PEDRA_NUM).fillna(0)
        grp2 = out.groupby("ra", sort=False)
        out["mudanca_pedra"] = grp2["pedra_num"].diff().fillna(0)
    return out


def engineered_row_for_display(df: pd.DataFrame, ra: str, ref_year: int | None = None) -> pd.Series | None:
    """Mesma linha (RA + ano) que `row_features_from_df`, mas após `ensure_risk_engineering` para UI e simulador.

    Ordem das colunas de ano alinhada a `_year_series_for_sort`: **Ano** primeiro, depois **ano_referencia**.
    Evita filtrar pelo ano «errado» quando as duas colunas divergem na base.
    """
    ra_col = "RA" if "RA" in df.columns else "ra"
    if ra_col not in df.columns:
        return None
    eng = ensure_risk_engineering(df)
    if ra_col not in eng.columns:
        return None
    cand = eng[eng[ra_col].astype(str) == str(ra)].copy()
    if cand.empty:
        return None

    if ref_year is not None:
        yr = float(int(ref_year))
        matched = None
        for ac in ("Ano", "ano_referencia"):
            if ac not in cand.columns:
                continue
            s = pd.to_numeric(cand[ac], errors="coerce")
            slice_ok = cand[(s - yr).abs() < 0.51]
            if not slice_ok.empty:
                matched = slice_ok
                break
        if matched is None or matched.empty:
            return None
        cand = matched
    else:
        sub_raw = df[df[ra_col].astype(str) == str(ra)]
        if sub_raw.empty:
            return None
        pick = pick_latest_year_row(sub_raw)
        if pick.empty:
            return None
        y_last = None
        for ac in ("Ano", "ano_referencia"):
            if ac in pick.columns and pd.notna(pick.iloc[0][ac]):
                try:
                    y_last = float(pick.iloc[0][ac])
                except (TypeError, ValueError):
                    y_last = None
                break
        if y_last is None:
            return cand.iloc[0]
        matched = None
        for ac in ("Ano", "ano_referencia"):
            if ac not in cand.columns:
                continue
            s = pd.to_numeric(cand[ac], errors="coerce")
            slice_ok = cand[(s - y_last).abs() < 0.51]
            if not slice_ok.empty:
                matched = slice_ok
                break
        if matched is not None and not matched.empty:
            cand = matched
    return cand.iloc[0]


def is_sklearn_risk_pipeline(bundle: Any) -> bool:
    return hasattr(bundle, "named_steps") and "pre" in getattr(bundle, "named_steps", {})


def _coerce_risk_categoricals_for_sklearn(d: pd.DataFrame) -> pd.DataFrame:
    """Evita dtype numérico em colunas categóricas (ex.: coluna só com NaN → float64).

    Caso contrário, o sklearn 1.4+ em `_check_unknown` trata `values` como numéricos e
    chama `isnan` nas categorias conhecidas (strings do treino) — TypeError.
    """
    out = d.copy()
    for c in RISK_CAT_FEATURES:
        if c not in out.columns:
            continue
        s = out[c]
        if pd.api.types.is_numeric_dtype(s):
            if c == "fase":
                out[c] = s.map(
                    lambda v: np.nan
                    if pd.isna(v)
                    else _fase_string_from_number(float(v))
                )
            else:
                out[c] = s.map(lambda v: np.nan if pd.isna(v) else str(v))
        out[c] = out[c].astype(object)
    return out


def risk_X_matrix(df: pd.DataFrame) -> pd.DataFrame:
    d = ensure_risk_engineering(df)
    missing = [c for c in RISK_MODEL_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para o modelo de risco: {missing}")
    d = _coerce_risk_categoricals_for_sklearn(d)
    return d[RISK_MODEL_COLUMNS]


def _year_numeric_series(df: pd.DataFrame) -> pd.Series:
    for c in ("Ano", "ano_referencia"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def risk_X_matrix_select_rows(full_df: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Linhas de `risk_X_matrix(full_df)` alinhadas a cada linha de `sub` (RA + ano), na ordem de `sub`.

    A engenharia (`media_turma_inde`, deltas por RA, etc.) corre sempre em **todo** `full_df`. Assim,
    a probabilidade de um aluno na **matriz** (recorte) coincide com a **ficha individual** no mesmo ano.
    """
    if sub.empty:
        return pd.DataFrame(columns=RISK_MODEL_COLUMNS)
    X_all = risk_X_matrix(full_df)
    ra_full = "RA" if "RA" in full_df.columns else "ra"
    ra_sub = "RA" if "RA" in sub.columns else "ra"
    y_full = _year_numeric_series(full_df)
    y_sub = _year_numeric_series(sub)
    parts: list[pd.DataFrame] = []
    for k in range(len(sub)):
        ra = str(sub.iloc[k][ra_sub])
        m = full_df[ra_full].astype(str) == ra
        yv = y_sub.iloc[k]
        if pd.notna(yv):
            m = m & (np.abs(y_full - float(yv)) < 0.51)
        hit = list(full_df.index[m])
        if not hit:
            Xi = risk_X_matrix(sub.iloc[[k]])
            parts.append(Xi)
            continue
        parts.append(X_all.loc[[hit[0]]])
    return pd.concat(parts, axis=0)


def row_matrix_by_ra(df: pd.DataFrame, ra: str) -> pd.DataFrame | None:
    key = "RA" if "RA" in df.columns else "ra"
    if key not in df.columns:
        return None
    sub = latest_single_row_for_ra(df, ra).copy()
    if sub.empty:
        return None
    return risk_X_matrix_select_rows(df, sub)


def row_matrix_for_ficha_feats(df: pd.DataFrame, feats: dict[str, Any]) -> pd.DataFrame | None:
    """Matriz de entrada do pipeline alinhada ao registo da ficha (RA + Ano em `feats`); fallback ao último ano."""
    key = "RA" if "RA" in df.columns else "ra"
    if key not in df.columns:
        return None
    ra = str(feats.get("RA", "")).strip()
    if not ra:
        return None
    ano_raw = feats.get("Ano")
    if ano_raw is None or pd.isna(ano_raw):
        return row_matrix_by_ra(df, ra)
    try:
        y = int(round(float(ano_raw)))
    except (TypeError, ValueError):
        return row_matrix_by_ra(df, ra)
    sub = single_row_for_ra_and_year(df, ra, y).copy()
    if sub.empty:
        return row_matrix_by_ra(df, ra)
    return risk_X_matrix_select_rows(df, sub)


def _sim_delta_inde(sim: dict[str, float]) -> float | None:
    """Valor explícito de delta_inde na simulação (Streamlit usa «Delta_INDE»)."""
    if "Delta_INDE" in sim:
        return float(sim["Delta_INDE"])
    if "delta_inde" in sim:
        return float(sim["delta_inde"])
    return None


def _apply_simulation_derived_row(sub: pd.DataFrame, idx: Any, media_turma_base: float, delta_inde_val: float) -> None:
    """Recalcula derivadas do notebook na linha simulada (evita valores obsoletos do Parquet)."""
    ida_v = float(sub.loc[idx, "ida"]) if "ida" in sub.columns else 0.0
    iaa_v = float(sub.loc[idx, "iaa"]) if "iaa" in sub.columns else 0.0
    inde_v = float(sub.loc[idx, "inde"]) if "inde" in sub.columns else 0.0
    ieg_v = float(sub.loc[idx, "ieg"]) if "ieg" in sub.columns else 0.0
    sub.loc[idx, "choque_realidade"] = iaa_v - ida_v
    if np.isnan(media_turma_base):
        sub.loc[idx, "distancia_media_turma"] = 0.0
    else:
        sub.loc[idx, "distancia_media_turma"] = inde_v - float(media_turma_base)
    sub.loc[idx, "delta_inde"] = float(delta_inde_val)
    sub.loc[idx, "esforco_sem_resultado"] = ieg_v / (ida_v + 0.1)


def build_X_after_slider_simulation(df: pd.DataFrame, ra: str, sim: dict[str, float]) -> pd.DataFrame | None:
    """Uma linha de entrada do pipeline após overrides do simulador (derivadas coerentes com o notebook)."""
    key = "RA" if "RA" in df.columns else "ra"
    if key not in df.columns:
        return None
    eng = ensure_risk_engineering(df.copy())
    # Alinhar ao ano da ficha na UI (`sim["Ano"]` == `row_features_from_df`); evitar misturar o último
    # registo do RA com o ano que o utilizador está a ver na matriz/ficha.
    resolved_year: int | None = None
    if "Ano" in sim:
        try:
            resolved_year = int(round(float(sim["Ano"])))
        except (TypeError, ValueError):
            resolved_year = None
    if resolved_year is not None:
        sub = single_row_for_ra_and_year(eng, ra, resolved_year).copy()
        if sub.empty:
            sub = latest_single_row_for_ra(eng, ra).copy()
    else:
        sub = latest_single_row_for_ra(eng, ra).copy()
    if sub.empty:
        return None
    idx = sub.index[0]

    media_turma_base = np.nan
    if "media_turma_inde" in sub.columns:
        v = sub.loc[idx, "media_turma_inde"]
        if pd.notna(v):
            media_turma_base = float(v)

    if "fase" in sub.columns and "Fase" in sim:
        fs = _coerce_fase_number(sim["Fase"])
        col = sub["fase"]
        if pd.api.types.is_numeric_dtype(col.dtype):
            sub.loc[idx, "fase"] = fs
        else:
            sub.loc[idx, "fase"] = _fase_string_from_number(fs)
    if "turma" in sub.columns and "Turma_ord" in sim:
        letter = _TURMA_ORD_LETTER.get(int(round(float(sim["Turma_ord"]))), "A")
        sub.loc[idx, "turma"] = letter
    if "ano_referencia" in sub.columns and "Ano" in sim:
        sub.loc[idx, "ano_referencia"] = int(round(float(sim["Ano"])))
    for up, lo in [
        ("INDE", "inde"),
        ("IDA", "ida"),
        ("IAN", "ian"),
        ("IEG", "ieg"),
        ("IPV", "ipv"),
        ("IAA", "iaa"),
        ("IPS", "ips"),
        ("IPP", "ipp"),
        ("ING", "ing"),
        ("MAT", "mat"),
        ("POR", "por"),
    ]:
        if up in sim and lo in sub.columns:
            sub.loc[idx, lo] = float(sim[up])
    if "pedra" in sub.columns and "Pedra_ord" in sim:
        sub.loc[idx, "pedra"] = _REV_PEDRA.get(int(round(float(sim["Pedra_ord"]))), sub.loc[idx, "pedra"])

    delta_explicit = _sim_delta_inde(sim)
    if delta_explicit is not None:
        delta_use = float(delta_explicit)
    else:
        prev = sub.loc[idx, "delta_inde"] if "delta_inde" in sub.columns else 0.0
        delta_use = float(prev) if pd.notna(prev) else 0.0

    _apply_simulation_derived_row(sub, idx, media_turma_base, delta_use)
    return risk_X_matrix(sub)
