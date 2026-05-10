"""Cenários numéricos pré-calculados para o parecer ML do Theo (sem UI de simulação)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from passos_magico.ml.features import FEATURE_ORDER
from passos_magico.ml.inference import predict_row_after_simulation
from passos_magico.ml.risk_pipeline import ensure_risk_engineering


def _eng_float(row: pd.Series | None, key: str, default: float) -> float:
    if row is None or key not in row.index:
        return default
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_baseline_sim_dict(feats: dict[str, Any], eng_row: pd.Series | None) -> dict[str, float]:
    """Alinha ao simulador antigo em `app/main.py` (FEATURE_ORDER + extras da linha engenharada)."""
    sim: dict[str, float] = {k: float(feats[k]) for k in FEATURE_ORDER}
    sim["IAA"] = _eng_float(eng_row, "iaa", 5.0)
    sim["IPS"] = _eng_float(eng_row, "ips", 5.0)
    sim["IPP"] = _eng_float(eng_row, "ipp", 5.0)
    sim["ING"] = _eng_float(eng_row, "ing", 5.0)
    sim["MAT"] = _eng_float(eng_row, "mat", 5.0)
    sim["POR"] = _eng_float(eng_row, "por", 5.0)
    sim["Delta_INDE"] = _eng_float(eng_row, "delta_inde", 0.0)
    return sim


def _pct_pt(p: float) -> str:
    return f"{p * 100:.1f}".replace(".", ",")


def _safe_predict(bundle: Any, df: pd.DataFrame, ra: str, sim: dict[str, float]) -> float | None:
    try:
        out = predict_row_after_simulation(bundle, df, ra, sim)
    except Exception:
        return None
    if out is None or (isinstance(out, float) and np.isnan(out)):
        return None
    return float(out)


def build_scenarios_markdown(
    bundle: Any,
    df: pd.DataFrame,
    ra: str,
    feats: dict[str, Any],
    eng_row: pd.Series | None,
    baseline_proba: float,
) -> str:
    """Bloco Markdown com percentagens já corridas pelo modelo; vazio se não houver bundle/df."""
    if bundle is None or df is None or df.empty or not feats:
        return ""
    sim0 = build_baseline_sim_dict(feats, eng_row)
    lines: list[str] = [
        "### Cenários já calculados pelo modelo (percentagens reais)",
        "Use **apenas** as percentagens desta lista para comparar cenários; **não** invente outras.",
        f"- **Linha de base (ficha na base):** {_pct_pt(float(baseline_proba))}%",
    ]
    seen: set[float] = {round(float(baseline_proba), 4)}

    def add_line(label: str, sim: dict[str, float]) -> None:
        p = _safe_predict(bundle, df, ra, sim)
        if p is None:
            return
        r = round(p, 4)
        if r in seen:
            return
        seen.add(r)
        lines.append(f"- {label}: {_pct_pt(p)}%")

    ieg = float(sim0["IEG"])
    if ieg < 9.99:
        s = dict(sim0)
        s["IEG"] = min(10.0, ieg + 1.0)
        add_line("Se **IEG** subir **+1** ponto (teto 10)", s)
    if ieg > 0.01:
        s = dict(sim0)
        s["IEG"] = max(0.0, ieg - 1.0)
        add_line("Se **IEG** descer **−1** ponto (mínimo 0)", s)

    inde = float(sim0["INDE"])
    if inde < 9.99:
        s = dict(sim0)
        s["INDE"] = min(10.0, inde + 0.5)
        add_line("Se **INDE** subir **+0,5** (teto 10)", s)
    if inde > 0.01:
        s = dict(sim0)
        s["INDE"] = max(0.0, inde - 0.5)
        add_line("Se **INDE** descer **−0,5** (mínimo 0)", s)

    ida = float(sim0["IDA"])
    if ida < 9.99:
        s = dict(sim0)
        s["IDA"] = min(10.0, ida + 0.5)
        add_line("Se **IDA** subir **+0,5** (teto 10)", s)

    iaa = float(sim0["IAA"])
    if abs(iaa - ida) > 0.05:
        s = dict(sim0)
        s["IAA"] = float(s["IDA"])
        add_line("Se **IAA** alinhar à **IDA** (autoavaliação = aprendizagem medida)", s)

    if len(lines) <= 3:
        return ""
    return "\n".join(lines)


def format_inde_history_summary(df: pd.DataFrame, ra: str) -> str:
    """Uma linha compacta de INDE por ano para o prompt (Theo interpretar tendência)."""
    ra_c = "RA" if "RA" in df.columns else "ra"
    if ra_c not in df.columns:
        return ""
    sub = df[df[ra_c].astype(str) == str(ra)].copy()
    if sub.empty:
        return ""
    try:
        eng = ensure_risk_engineering(sub)
    except Exception:
        return ""
    if "inde" not in eng.columns:
        return ""
    ycol = "ano_referencia" if "ano_referencia" in eng.columns else ("Ano" if "Ano" in eng.columns else None)
    if ycol is None:
        return ""
    eng = eng.copy()
    eng["_y"] = pd.to_numeric(eng[ycol], errors="coerce")
    eng = eng.dropna(subset=["_y", "inde"]).sort_values("_y")
    if eng.empty:
        return ""
    parts: list[str] = []
    for _, r in eng.iterrows():
        y = int(round(float(r["_y"])))
        ind = float(r["inde"])
        mt = r.get("media_turma_inde")
        if mt is not None and not (isinstance(mt, float) and np.isnan(mt)):
            parts.append(f"{y}: INDE aluno {ind:.1f}, média grupo {float(mt):.1f}")
        else:
            parts.append(f"{y}: INDE aluno {ind:.1f}")
    if not parts:
        return ""
    return "Evolução resumida na base (INDE): " + "; ".join(parts) + "."
