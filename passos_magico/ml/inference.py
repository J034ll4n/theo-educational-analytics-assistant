"""Carregamento do modelo, predição e SHAP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from passos_magico.data_engine.loader import PROJECT_ROOT
from passos_magico.ml.features import FEATURE_ORDER, augment_dataframe
from passos_magico.ml.risk_pipeline import (
    build_X_after_slider_simulation,
    is_sklearn_risk_pipeline,
    risk_X_matrix,
    risk_X_matrix_select_rows,
    row_matrix_for_ficha_feats,
)

PRIMARY_MODEL_PATH = PROJECT_ROOT / "modelo_risco_aluno.pkl"


def load_model_bundle(path: Path | None = None) -> Any:
    p = path if path is not None else PRIMARY_MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {p}. Treine com o notebook "
            f"`notebooks/Ml/ML_Passos_Magicos.ipynb` e grave `modelo_risco_aluno.pkl` na raiz do projeto."
        )
    return joblib.load(p)


def _get_clf(bundle: dict | Any):
    if isinstance(bundle, dict):
        if "clf" in bundle:
            return bundle["clf"]
        return bundle["model"]
    if is_sklearn_risk_pipeline(bundle):
        return bundle.named_steps["clf"]
    raise TypeError("Bundle de modelo não reconhecido.")


def _x_df_legacy(feats: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                feats["Fase"],
                feats["Turma_ord"],
                feats["Ano"],
                feats["INDE"],
                feats["IDA"],
                feats["IAN"],
                feats["IEG"],
                feats["IPV"],
                feats["Pedra_ord"],
            ]
        ],
        columns=FEATURE_ORDER,
    )


def predict_row_features(bundle: Any, feats: dict[str, float], df: pd.DataFrame | None = None) -> float:
    if is_sklearn_risk_pipeline(bundle):
        if df is None:
            raise ValueError("predict_row_features com pipeline XGBoost requer o DataFrame de dados (`df`).")
        ra = str(feats.get("RA", "")).strip()
        if not ra:
            raise ValueError("RA ausente em feats para predição com pipeline de risco.")
        xm = row_matrix_for_ficha_feats(df, feats)
        if xm is None or xm.empty:
            return float("nan")
        return float(bundle.predict_proba(xm)[0, 1])
    clf = _get_clf(bundle)
    row_feats = {k: float(feats[k]) for k in FEATURE_ORDER}
    X_df = _x_df_legacy(row_feats)
    return float(clf.predict_proba(X_df)[0, 1])


def predict_risk_probabilities(bundle: Any, df: pd.DataFrame) -> np.ndarray:
    """Probabilidade P(alto risco) por linha, na mesma ordem de `df`."""
    if df.empty:
        return np.array([], dtype=np.float64)
    if is_sklearn_risk_pipeline(bundle):
        X = risk_X_matrix(df)
        return bundle.predict_proba(X)[:, 1].astype(np.float64)
    d = augment_dataframe(df.copy())
    clf = _get_clf(bundle)
    X = d[FEATURE_ORDER].astype(float)
    return clf.predict_proba(X)[:, 1].astype(np.float64)


def ensure_risco_column(df: pd.DataFrame, bundle: Any | None) -> pd.DataFrame:
    """Garante coluna `risco` (0–1) alinhada ao modelo quando ainda não existir no DataFrame."""
    if df.empty or bundle is None or "risco" in df.columns:
        return df
    try:
        probs = predict_risk_probabilities(bundle, df)
        out = df.copy()
        out["risco"] = probs
        return out
    except Exception:
        return df


def predict_risk_slice(bundle: Any, sub: pd.DataFrame, full_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Probabilidade de risco por linha de `sub` (já recortado), na ordem do modelo; ordena do maior para o menor.

    Com pipeline sklearn, passe **`full_df`** = base completa carregada na app: a engenharia de features
    (média da turma, deltas temporais, etc.) fica igual à ficha individual. Sem `full_df`, mantém-se o
    comportamento antigo (só o recorte), útil para testes mínimos.
    """
    if sub.empty:
        return pd.DataFrame(columns=["RA", "Nome", "Fase", "Turma", "Ano", "risco"])
    sub = sub.copy()
    ra_col = "RA" if "RA" in sub.columns else "ra"
    nome_col = "Nome" if "Nome" in sub.columns else "nome"
    f_col = "Fase" if "Fase" in sub.columns else "fase"
    t_col = "Turma" if "Turma" in sub.columns else "turma"
    ano_col = "Ano" if "Ano" in sub.columns else "ano_referencia"
    if is_sklearn_risk_pipeline(bundle):
        if full_df is not None:
            X = risk_X_matrix_select_rows(full_df, sub)
        else:
            X = risk_X_matrix(sub)
        prob = bundle.predict_proba(X)[:, 1]
    else:
        clf = _get_clf(bundle)
        d = augment_dataframe(sub)
        X = d[FEATURE_ORDER].astype(float)
        prob = clf.predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            "RA": sub[ra_col].astype(str),
            "Nome": sub[nome_col].astype(str) if nome_col in sub.columns else "",
            "Fase": sub[f_col],
            "Turma": sub[t_col] if t_col in sub.columns else "",
            "Ano": sub[ano_col] if ano_col in sub.columns else pd.NA,
            "risco": prob,
        }
    )
    return out.sort_values("risco", ascending=False)


def predict_risk_batch(bundle: Any, df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return predict_risk_slice(bundle, df.loc[mask].copy(), full_df=df)


def explain_row_shap(bundle: Any, feats: dict[str, float], df: pd.DataFrame | None = None) -> list[tuple[str, float]]:
    if is_sklearn_risk_pipeline(bundle):
        if df is None:
            return []
        ra = str(feats.get("RA", "")).strip()
        xm = row_matrix_for_ficha_feats(df, feats) if ra else None
        if xm is None or xm.empty:
            return []
        pre = bundle.named_steps["pre"]
        clf = bundle.named_steps["clf"]
        Xt = pre.transform(xm)
        names = list(pre.get_feature_names_out())
        names = [n.split("__")[-1] for n in names]
        try:
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(Xt)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            sv = np.asarray(shap_vals).reshape(-1)
            if sv.size != len(names):
                sv = np.asarray(shap_vals[0]).reshape(-1)
            pairs = list(zip(names, [float(x) for x in sv], strict=False))
        except Exception:
            imp = getattr(clf, "feature_importances_", np.ones(len(names)) / len(names))
            pairs = list(zip(names, [float(x) for x in imp], strict=False))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs
    clf = _get_clf(bundle)
    row_feats = {k: float(feats[k]) for k in FEATURE_ORDER}
    X_df = _x_df_legacy(row_feats)
    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        sv = np.asarray(shap_vals).reshape(-1)
        if sv.size != len(FEATURE_ORDER):
            sv = np.asarray(shap_vals[0]).reshape(-1)
        pairs = list(zip(FEATURE_ORDER, [float(x) for x in sv], strict=True))
    except Exception:
        imp = clf.feature_importances_
        pairs = list(zip(FEATURE_ORDER, [float(x) for x in imp], strict=True))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs


def predict_row_after_simulation(bundle: Any, df: pd.DataFrame, ra: str, sim: dict[str, float]) -> float:
    if not is_sklearn_risk_pipeline(bundle):
        clf = _get_clf(bundle)
        row_feats = {k: float(sim[k]) for k in FEATURE_ORDER}
        X_df = _x_df_legacy(row_feats)
        return float(clf.predict_proba(X_df)[0, 1])
    xm = build_X_after_slider_simulation(df, ra, sim)
    if xm is None:
        return float("nan")
    return float(bundle.predict_proba(xm)[0, 1])
