"""Heurísticas de qualidade da ficha para contexto ML (sem dependência de LLM)."""

from __future__ import annotations

import math
from typing import Any


def is_zero_pedagogy_score(v: Any) -> bool:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return not math.isnan(x) and abs(x) < 1e-9


def ficha_suspect_missing_indicators(feats: dict[str, Any] | None) -> bool:
    """True se zeros em IDA/IEG/IPV parecem lacuna (INDE preenchido na escala típica 0–10)."""
    if not feats:
        return False
    try:
        inde = float(feats.get("INDE", float("nan")))
    except (TypeError, ValueError):
        inde = float("nan")
    if not (inde == inde) or not (0 < inde < 10):
        return False
    z_ida = is_zero_pedagogy_score(feats.get("IDA"))
    z_ieg = is_zero_pedagogy_score(feats.get("IEG"))
    z_ipv = is_zero_pedagogy_score(feats.get("IPV"))
    n_zero = sum(1 for z in (z_ida, z_ieg, z_ipv) if z)
    if z_ida and z_ieg:
        return True
    return n_zero >= 2


def ficha_quality_snapshot_extra_lines(feats: dict[str, Any] | None) -> list[str]:
    """Linhas Markdown a acrescentar ao snapshot da ficha quando houver suspeita de lacuna."""
    if not ficha_suspect_missing_indicators(feats):
        return []
    return [
        "",
        "Qualidade dos dados na ficha:",
        (
            "- Vários indicadores (IDA, IEG e/ou IPV) estão em **0** com INDE preenchido; "
            "suspeita forte de **lacuna no relatório ou dado não consolidado**, não de desempenho real "
            "«zero» na escala 0–10."
        ),
        (
            "- **Não** interprete esses zeros como participação ou aprendizagem «normais»; "
            "priorize indicadores consistentes (ex.: INDE, IAN) e o SHAP, e confirme com a equipe na base bruta."
        ),
    ]
