"""Carregamento do modelo, predição e SHAP."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from passos_magico.data_engine.loader import PROJECT_ROOT
from passos_magico.ml.features import FEATURE_ORDER, augment_dataframe

MODEL_PATH = PROJECT_ROOT / "models" / "modelo.joblib"


def load_model_bundle(path: Path | None = None) -> dict:
    p = path or MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {p}. Execute scripts/train_model.py."
        )
    bundle = joblib.load(p)
    return bundle


def _get_clf(bundle: dict):
    if "clf" in bundle:
        return bundle["clf"]
    return bundle["model"]


def _x_df(feats: dict[str, float]) -> pd.DataFrame:
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


def predict_row_features(bundle: dict, feats: dict[str, float]) -> float:
    clf = _get_clf(bundle)
    X_df = _x_df(feats)
    proba = clf.predict_proba(X_df)[0, 1]
    return float(proba)


def predict_risk_probabilities(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Probabilidade P(alto risco) por linha, na mesma ordem de `df`."""
    d = augment_dataframe(df.copy())
    if d.empty:
        return np.array([], dtype=np.float64)
    clf = _get_clf(bundle)
    X = d[FEATURE_ORDER].astype(float)
    return clf.predict_proba(X)[:, 1].astype(np.float64)


def predict_risk_batch(bundle: dict, df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    clf = _get_clf(bundle)
    d = augment_dataframe(df.loc[mask].copy())
    if d.empty:
        return pd.DataFrame(columns=["RA", "Nome", "Fase", "Turma", "Ano", "risco"])
    X = d[FEATURE_ORDER].astype(float)
    prob = clf.predict_proba(X)[:, 1]
    out = d[["RA", "Nome", "Fase", "Turma", "Ano"]].copy()
    out["risco"] = prob
    return out.sort_values("risco", ascending=False)


def explain_row_shap(bundle: dict, feats: dict[str, float]) -> list[tuple[str, float]]:
    """Valores SHAP (TreeExplainer) por feature; fallback em importâncias."""
    clf = _get_clf(bundle)
    X_df = _x_df(feats)
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
