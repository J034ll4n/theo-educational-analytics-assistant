"""Mapa de conceitos curado — ligações pedagógicas entre indicadores (não são FKs na base)."""

from __future__ import annotations

import math
from typing import Iterable

import plotly.graph_objects as go

# (origem, destino, texto para legenda)
CURATED_EDGES: tuple[tuple[str, str, str], ...] = (
    ("INDE", "IDA", "INDE e IDA: visão global do desenvolvimento e desempenho em aprendizagem."),
    ("INDE", "IAN", "INDE e IAN: resultado agregado e adequação ao nível esperado."),
    ("IDA", "IAN", "IDA e IAN: aprendizagem medida e adequação ao nível."),
    ("IEG", "IDA", "IEG e IDA: engajamento associado ao desempenho acadêmico."),
    ("IEG", "INDE", "IEG e INDE: engajamento e desenvolvimento educacional global."),
    ("IPV", "IDA", "IPV e IDA: traço de ponto de virada e aprendizagem."),
    ("IPS", "INDE", "IPS e INDE: fatores psicossociais e visão global (quando IPS existir na base)."),
)


def edges_visible(columns: Iterable[str]) -> list[tuple[str, str, str]]:
    colset = {str(c).strip() for c in columns if c is not None and str(c).strip()}
    return [(a, b, d) for a, b, d in CURATED_EDGES if a in colset and b in colset]


def build_concept_map_figure(columns: Iterable[str]) -> go.Figure | None:
    """Grafo simples (Plotly) só com arestas curadas cujos dois nós existem no dicionário."""
    edges = edges_visible(columns)
    nodes = sorted({n for e in edges for n in (e[0], e[1])})
    if len(nodes) < 2:
        return None

    n = len(nodes)
    pos: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        ang = 2 * math.pi * i / n - math.pi / 2
        pos[node] = (math.cos(ang), math.sin(ang))

    accent = "#EE145B"
    fig = go.Figure()
    for a, b, _ in edges:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="rgba(238,20,91,0.5)", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[pos[node][0] for node in nodes],
            y=[pos[node][1] for node in nodes],
            mode="markers+text",
            text=nodes,
            textfont=dict(color="#f0f3f6", size=13),
            textposition="top center",
            marker=dict(size=28, color="#30363d", line=dict(color=accent, width=1)),
            hoverinfo="text",
            hovertext=nodes,
            showlegend=False,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,36,0.9)",
        font=dict(color="#e6edf3", size=12),
        title=dict(text="Indicadores ligados (resumo)", font=dict(size=14), x=0, xanchor="left"),
        margin=dict(l=32, r=32, t=44, b=32),
        height=340,
        xaxis=dict(
            visible=False,
            range=[-1.35, 1.35],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(visible=False, range=[-1.35, 1.35]),
        showlegend=False,
    )
    return fig
