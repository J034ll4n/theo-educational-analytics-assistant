"""Geração heurística de gráficos Plotly a partir de DataFrames."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# IDs estáveis para o seletor na UI (ordem de exibição)
CHART_TYPE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Automático (mesma lógica do Theo)"),
    ("linha", "Linhas"),
    ("barras_v", "Barras verticais"),
    ("barras_h", "Barras horizontais"),
    ("scatter", "Dispersão"),
    ("area", "Área"),
    ("histograma", "Histograma"),
    ("pizza", "Pizza"),
    ("box", "Caixa (boxplot)"),
]

# Mapeia rótulo heurístico antigo → id preferido no seletor
HEURISTIC_TO_CHART_ID: dict[str, str] = {
    "vazio": "auto",
    "linha": "linha",
    "barras": "barras_v",
    "histograma": "histograma",
    "barras (contagem)": "barras_v",
    "erro": "auto",
    "scatter": "scatter",
    "pizza": "pizza",
    "area": "area",
    "box": "box",
}


def _narrow_df(df: pd.DataFrame, max_cols: int = 12) -> pd.DataFrame:
    out = df.copy()
    n = min(len(out.columns), max_cols)
    return out.iloc[:, :n]


def _sanitize_df_for_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas duplicadas e colunas numéricas redundantes (mesmos valores),
    cenário comum quando o SQL projeta a mesma medida duas vezes — evita gráfico
    de linha/barras comparando a coluna com ela mesma.
    """
    if df.empty or len(df.columns) == 0:
        return df
    d = df.copy()
    # Nomes de coluna duplicados no DataFrame (pandas permite)
    d = d.loc[:, ~d.columns.duplicated()].copy()
    num_names = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    drop: set[str] = set()
    kept: list[str] = []
    for c in num_names:
        redundant = False
        for prev in kept:
            try:
                if d[c].equals(d[prev]):
                    redundant = True
                    break
                if len(d[c]) == len(d[prev]) and (d[c].fillna(0).values == d[prev].fillna(0).values).all():
                    redundant = True
                    break
            except Exception:
                pass
        if redundant:
            drop.add(c)
        else:
            kept.append(c)
    if drop:
        d = d.drop(columns=[x for x in drop if x in d.columns], errors="ignore")
    return d


def _column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols = list(df.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#8b949e"),
            )
        ],
    )
    return fig


def dataframe_to_figure(df: pd.DataFrame) -> tuple[go.Figure, str]:
    """Retorna figura Plotly e descrição curta do tipo de gráfico (heurística automática)."""
    return figure_from_dataframe(df, "auto")


def figure_from_dataframe(df: pd.DataFrame, chart_id: str) -> tuple[go.Figure, str]:
    """
    Monta uma figura Plotly a partir do DataFrame e do tipo escolhido.
    `chart_id` usa os IDs de CHART_TYPE_OPTIONS; "auto" replica a heurística original.
    """
    if df.empty:
        return _empty_fig("Nenhum dado retornado."), "vazio"

    d = _sanitize_df_for_chart(_narrow_df(df))
    if len(d.columns) == 0:
        return _empty_fig("Nenhuma coluna útil após limpar dados duplicados."), "vazio"
    num_cols, cat_cols = _column_groups(d)

    if chart_id == "auto":
        # Barras (categoria + valor) primeiro: comparações público/particular, turmas, etc.
        if len(cat_cols) >= 1 and len(num_cols) >= 1:
            fig = px.bar(
                d.head(100),
                x=cat_cols[0],
                y=num_cols[0],
                template="plotly_dark",
            )
            fig.update_layout(xaxis_title=cat_cols[0], yaxis_title=num_cols[0])
            return fig, "barras"
        # Duas séries numéricas (ex.: x temporal e y métrica) — linha
        if len(num_cols) >= 2 and len(d) <= 500:
            fig = px.line(
                d,
                x=num_cols[0],
                y=num_cols[1],
                markers=True,
                template="plotly_dark",
            )
            return fig, "linha"
        if len(num_cols) >= 1:
            fig = px.histogram(
                d,
                x=num_cols[0],
                template="plotly_dark",
                nbins=min(40, max(10, len(d) // 3)),
            )
            return fig, "histograma"
        c0 = list(d.columns)[0]
        vc = d[c0].astype(str).value_counts().head(30)
        fig = px.bar(
            x=vc.index,
            y=vc.values,
            labels={"x": c0, "y": "contagem"},
            template="plotly_dark",
        )
        return fig, "barras (contagem)"  # legado para prompts

    try:
        if chart_id == "linha":
            if len(num_cols) >= 2:
                fig = px.line(
                    d,
                    x=num_cols[0],
                    y=num_cols[1],
                    markers=True,
                    template="plotly_dark",
                )
            elif len(cat_cols) >= 1 and len(num_cols) >= 1:
                fig = px.line(
                    d.sort_values(cat_cols[0]),
                    x=cat_cols[0],
                    y=num_cols[0],
                    markers=True,
                    template="plotly_dark",
                )
            else:
                return (
                    _empty_fig("Linhas precisam de pelo menos 2 colunas numéricas, ou 1 categoria + 1 número."),
                    "erro",
                )
            return fig, "linha"

        if chart_id == "barras_v":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                fig = px.bar(
                    d.head(200),
                    x=cat_cols[0],
                    y=num_cols[0],
                    template="plotly_dark",
                )
                fig.update_layout(xaxis_title=cat_cols[0], yaxis_title=num_cols[0])
            elif len(num_cols) >= 1:
                agg = d.groupby(num_cols[0], as_index=False).size()
                fig = px.bar(agg, x=num_cols[0], y="size", template="plotly_dark")
            else:
                c0 = list(d.columns)[0]
                vc = d[c0].astype(str).value_counts().head(40)
                fig = px.bar(x=vc.index, y=vc.values, template="plotly_dark")
            return fig, "barras"

        if chart_id == "barras_h":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                fig = px.bar(
                    d.head(200),
                    x=num_cols[0],
                    y=cat_cols[0],
                    orientation="h",
                    template="plotly_dark",
                )
            elif len(num_cols) >= 1:
                vc = d[num_cols[0]].value_counts().head(40)
                fig = px.bar(x=vc.values, y=vc.index.astype(str), orientation="h", template="plotly_dark")
            else:
                c0 = list(d.columns)[0]
                vc = d[c0].astype(str).value_counts().head(40)
                fig = px.bar(x=vc.values, y=vc.index.astype(str), orientation="h", template="plotly_dark")
            return fig, "barras"

        if chart_id == "scatter":
            if len(num_cols) >= 2:
                fig = px.scatter(d, x=num_cols[0], y=num_cols[1], template="plotly_dark")
            elif len(cat_cols) >= 1 and len(num_cols) >= 1:
                fig = px.scatter(d, x=cat_cols[0], y=num_cols[0], template="plotly_dark")
            else:
                return _empty_fig("Dispersão precisa de pelo menos duas colunas numéricas."), "erro"
            return fig, "scatter"

        if chart_id == "area":
            if len(num_cols) >= 2:
                tmp = d.sort_values(num_cols[0])
                fig = px.area(tmp, x=num_cols[0], y=num_cols[1], template="plotly_dark")
            elif len(cat_cols) >= 1 and len(num_cols) >= 1:
                tmp = d.sort_values(cat_cols[0])
                fig = px.area(tmp, x=cat_cols[0], y=num_cols[0], template="plotly_dark")
            else:
                return _empty_fig("Área precisa de eixo X e Y numéricos ou categoria + número."), "erro"
            return fig, "area"

        if chart_id == "histograma":
            if len(num_cols) >= 1:
                fig = px.histogram(
                    d,
                    x=num_cols[0],
                    template="plotly_dark",
                    nbins=min(50, max(10, len(d) // 3)),
                )
            else:
                c0 = list(d.columns)[0]
                fig = px.histogram(d, x=c0, template="plotly_dark")
            return fig, "histograma"

        if chart_id == "pizza":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                pie_df = d.groupby(cat_cols[0], as_index=False)[num_cols[0]].sum()
                fig = px.pie(pie_df, names=cat_cols[0], values=num_cols[0], template="plotly_dark")
            elif len(cat_cols) >= 1:
                vc = d[cat_cols[0]].astype(str).value_counts().head(20)
                fig = px.pie(names=vc.index, values=vc.values, template="plotly_dark")
            else:
                return _empty_fig("Pizza precisa de uma coluna de categoria (e opcionalmente valores numéricos)."), "erro"
            return fig, "pizza"

        if chart_id == "box":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                fig = px.box(d, x=cat_cols[0], y=num_cols[0], template="plotly_dark")
            elif len(num_cols) >= 1:
                fig = px.box(d, y=num_cols[0], template="plotly_dark")
            else:
                return _empty_fig("Boxplot precisa de pelo menos uma coluna numérica."), "erro"
            return fig, "box"
    except Exception:
        return (
            _empty_fig("Não foi possível montar este tipo de gráfico com estes dados. Tente outro."),
            "erro",
        )

    return figure_from_dataframe(df, "auto")


def heuristic_kind_to_chart_id(kind: str) -> str:
    """Converte o rótulo heurístico salvo no turno para id do seletor."""
    return HEURISTIC_TO_CHART_ID.get(kind, "auto")
