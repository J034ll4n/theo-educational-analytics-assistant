"""Define se o texto do Theo deve ser curto (KPI) ou analítico."""

from __future__ import annotations

import pandas as pd


def infer_insight_response_mode(df: pd.DataFrame | None, chart_kind: str) -> str:
    """
    Retorna ``kpi`` ou ``analitico`` (rótulo enviado ao LLM no user message).

    - ``chart_kind == "kpi"``: resultado escalar sem gráfico útil.
    - ``len(df) == 1``: uma linha (contagem agregada, média única, etc.).
    """
    if chart_kind == "kpi":
        return "kpi"
    if df is not None and not df.empty and len(df) == 1:
        return "kpi"
    return "analitico"
