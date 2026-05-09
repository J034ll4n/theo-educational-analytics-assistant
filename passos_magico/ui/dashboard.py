"""Dashboards analíticos sobre o Parquet — KPIs e gráficos simples."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from passos_magico.ml.inference import predict_risk_probabilities
from passos_magico.ml.risk_display import OPERATIONAL_HIGH_RISK_THRESHOLD
from passos_magico.ui.dashboard_catalog import IAN_ADEQUACAO, INDICATOR_CATALOG
from passos_magico.ui.dashboard_feedback import render_dashboard_theo_feedback

# Paleta alinhada ao fundo escuro da app (.stApp ≈ #121218)
_DASH_ACCENT = "#EE145B"
_DASH_ACCENT2 = "#00d4d8"
_DASH_TEXT = "#e6edf3"
_DASH_MUTED = "#94a3b8"
_DASH_GRID = "rgba(255,255,255,0.08)"
_DASH_PAPER = "#121218"
_DASH_PLOT_BG = "#16161d"

RISKO_LIMIAR: float = OPERATIONAL_HIGH_RISK_THRESHOLD


def _ensure_risco(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if "risco" in out.columns and out["risco"].notna().any():
        return out
    try:
        out["risco"] = predict_risk_probabilities(bundle, out)
    except Exception:
        out["risco"] = np.nan
    return out


def _numeric_indicators_in_df(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for code, _ in INDICATOR_CATALOG:
        if code not in df.columns:
            continue
        s = df[code]
        if pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors="coerce").notna().any():
            out.append(code)
    return out


def _dash_figure_layout(
    fig: go.Figure,
    title: str,
    *,
    height: int = 380,
    calendar_year_x: bool = False,
    calendar_year_y: bool = False,
) -> go.Figure:
    """Gráficos com fundo escuro, alinhados à página Streamlit."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DASH_PAPER,
        plot_bgcolor=_DASH_PLOT_BG,
        font=dict(color=_DASH_TEXT, size=12, family="Inter, Segoe UI, system-ui, sans-serif"),
        title=dict(text=title, font=dict(size=15, color=_DASH_TEXT), x=0.02, xanchor="left"),
        margin=dict(l=52, r=28, t=56, b=48),
        height=height,
        colorway=[_DASH_ACCENT, _DASH_ACCENT2, "#79c0ff", "#fbbf24"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(22, 24, 30, 0.92)",
            bordercolor="rgba(238, 20, 91, 0.35)",
            borderwidth=1,
            font=dict(color=_DASH_TEXT),
        ),
    )
    fig.update_xaxes(
        gridcolor=_DASH_GRID,
        zeroline=False,
        linecolor="rgba(148, 163, 184, 0.35)",
        tickfont=dict(color=_DASH_MUTED),
        title_font=dict(color=_DASH_TEXT),
    )
    fig.update_yaxes(
        gridcolor=_DASH_GRID,
        zeroline=False,
        linecolor="rgba(148, 163, 184, 0.35)",
        tickfont=dict(color=_DASH_MUTED),
        title_font=dict(color=_DASH_TEXT),
    )
    fig.update_annotations(font=dict(color=_DASH_TEXT))
    tick = dict(tickformat=".0f", dtick=1, separatethousands=False)
    if calendar_year_x:
        fig.update_xaxes(**tick)
    if calendar_year_y:
        fig.update_yaxes(**tick)
    return fig


def _kpi_row(df: pd.DataFrame) -> None:
    n_uni = df["RA"].nunique() if "RA" in df.columns else len(df)
    m_inde = float(df["INDE"].mean()) if "INDE" in df.columns else float("nan")
    m_ida = float(df["IDA"].mean()) if "IDA" in df.columns else float("nan")
    if "IAN" in df.columns:
        pct_ok = float((pd.to_numeric(df["IAN"], errors="coerce") >= IAN_ADEQUACAO).mean() * 100)
    else:
        pct_ok = float("nan")
    if "risco" in df.columns and df["risco"].notna().any():
        pct_alto = float((df["risco"] >= RISKO_LIMIAR).mean() * 100)
    else:
        pct_alto = float("nan")

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        st.metric("Alunos (únicos)", f"{n_uni:,}".replace(",", "."))
    with c2:
        st.metric("INDE médio", f"{m_inde:.2f}" if m_inde == m_inde else "—")
    with c3:
        st.metric("IDA médio", f"{m_ida:.2f}" if m_ida == m_ida else "—")
    with c4:
        st.metric(f"IAN ≥ {IAN_ADEQUACAO:g}", f"{pct_ok:.1f}%" if pct_ok == pct_ok else "—")
    with c5:
        if pct_alto == pct_alto:
            _lim = int(RISKO_LIMIAR * 100)
            st.metric(
                label="Alunos acima do corte (modelo)",
                value=f"{pct_alto:.1f}%",
                help=(
                    f"**O que conta aqui:** **{pct_alto:.1f}%** dos alunos **deste filtro** têm probabilidade de alto risco **igual ou superior a {_lim}%** "
                    f"(é esse o **corte** usado no painel). O **{_lim}%** não substitui o **{pct_alto:.1f}%** — um é o patamar, o outro é a **fatia** do grupo. "
                    "No gráfico «Distribuição de risco», o corte aparece como linha tracejada."
                ),
            )
        else:
            st.metric("Risco ML (sem dados)", "—")


def _chart_ian_por_ano(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["IAN", "Ano"]).copy()
    d["Ano"] = pd.to_numeric(d["Ano"], errors="coerce")
    d = d.dropna(subset=["Ano"])
    fig = px.box(d, x="Ano", y="IAN", points="outliers", labels={"IAN": "IAN", "Ano": "Ano"})
    fig.update_traces(boxmean="sd")
    return _dash_figure_layout(fig, "IAN por ano", calendar_year_x=True)


def _chart_base_por_ano(df: pd.DataFrame) -> go.Figure:
    if "RA" in df.columns:
        c = df.groupby("Ano")["RA"].nunique().reset_index(name="alunos")
    else:
        c = df.groupby("Ano").size().reset_index(name="alunos")
    fig = px.bar(c, x="Ano", y="alunos", labels={"alunos": "Alunos", "Ano": "Ano"})
    fig.update_traces(marker_color=_DASH_ACCENT, marker_line_width=0, opacity=0.9)
    return _dash_figure_layout(fig, "Alunos por ano", calendar_year_x=True)


def _chart_inde_pedra(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["INDE", "Pedra"])
    g = d.groupby("Pedra", observed=True)["INDE"].mean().reset_index()
    fig = px.bar(g, x="Pedra", y="INDE", labels={"INDE": "INDE médio", "Pedra": "Pedra"})
    fig.update_traces(marker_color=_DASH_ACCENT2, marker_line_width=0, opacity=0.9)
    return _dash_figure_layout(fig, "INDE médio por Pedra")


def _chart_correlacao(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in ("INDE", "IDA", "IAN", "IEG", "IPV") if c in df.columns]
    if len(cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Dados insuficientes", showarrow=False)
        return _dash_figure_layout(fig, "Correlação entre indicadores", height=300)
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = num.corr().fillna(0)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_traces(hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>")
    return _dash_figure_layout(fig, "Correlação (Pearson)")


def _chart_risco_dist(df: pd.DataFrame) -> go.Figure:
    if "risco" not in df.columns or df["risco"].isna().all():
        fig = go.Figure()
        fig.add_annotation(text="Risco indisponível", showarrow=False)
        return _dash_figure_layout(fig, "Distribuição de risco (modelo)")
    fig = px.histogram(df, x="risco", nbins=36, labels={"risco": "Probabilidade", "count": "N"})
    fig.update_layout(showlegend=False)
    fig.update_traces(marker_color=_DASH_ACCENT, marker_line_color="rgba(255,255,255,0.15)", marker_line_width=0.5, opacity=0.88)
    fig.add_vline(
        x=RISKO_LIMIAR,
        line_dash="dash",
        line_width=2,
        line_color=_DASH_ACCENT2,
        annotation_text=f" Limiar {int(RISKO_LIMIAR * 100)}%",
        annotation_position="top",
        annotation_font_color=_DASH_TEXT,
        annotation_bgcolor="rgba(18,18,24,0.85)",
    )
    return _dash_figure_layout(fig, "Distribuição de risco")


def _indicator_label(code: str) -> str:
    for c, title in INDICATOR_CATALOG:
        if c == code:
            return title
    return code


def _charts_for_indicator(df: pd.DataFrame, ind: str) -> None:
    st.subheader(f"{ind} — {_indicator_label(ind)}")
    y = pd.to_numeric(df[ind], errors="coerce")
    base = df.assign(_y=y).dropna(subset=["_y", "Ano"])

    c1, c2 = st.columns(2, gap="small")
    with c1:
        fig1 = px.box(base, x="Ano", y="_y", labels={"_y": ind, "Ano": "Ano"})
        fig1.for_each_trace(lambda t: t.update(showlegend=False))
        fig1.update_traces(boxmean="sd")
        _dash_figure_layout(fig1, f"{ind} por ano", calendar_year_x=True)
        st.plotly_chart(fig1, width="stretch", key=f"dbx_{ind}_ano")

    with c2:
        if ind != "INDE" and "INDE" in df.columns:
            sc = df.assign(_y=y, xi=pd.to_numeric(df["INDE"], errors="coerce")).dropna(subset=["_y", "xi"])
            fig2 = px.scatter(sc, x="xi", y="_y", labels={"xi": "INDE", "_y": ind}, opacity=0.65)
            ttl = f"{ind} × INDE"
        elif "INDE" in df.columns and "IDA" in df.columns:
            sc = df.assign(
                xi=pd.to_numeric(df["INDE"], errors="coerce"),
                yi=pd.to_numeric(df["IDA"], errors="coerce"),
            ).dropna(subset=["xi", "yi"])
            fig2 = px.scatter(sc, x="xi", y="yi", labels={"xi": "INDE", "yi": "IDA"}, opacity=0.65)
            ttl = "INDE × IDA"
        else:
            fig2 = go.Figure()
            fig2.add_annotation(text="Sem dados para dispersão", showarrow=False)
            ttl = "Dispersão"
        _dash_figure_layout(fig2, ttl)
        st.plotly_chart(fig2, width="stretch", key=f"dbx_{ind}_scatter")

    c3, c4 = st.columns(2, gap="small")
    with c3:
        if "Fase" in df.columns:
            g = (
                df.assign(_y=y)
                .dropna(subset=["_y", "Fase"])
                .groupby("Fase", observed=True)["_y"]
                .mean()
                .reset_index()
                .sort_values("Fase")
            )
            fig3 = px.bar(g, x="Fase", y="_y", labels={"_y": f"Média {ind}", "Fase": "Fase"})
            fig3.update_traces(marker_color=_DASH_ACCENT2, marker_line_width=0, opacity=0.9)
            _dash_figure_layout(fig3, f"{ind} — média por fase")
        else:
            fig3 = go.Figure()
            fig3.add_annotation(text="Sem coluna Fase", showarrow=False)
            _dash_figure_layout(fig3, f"{ind} por fase")
        st.plotly_chart(fig3, width="stretch", key=f"dbx_{ind}_fase")

    with c4:
        ev = (
            df.assign(_y=y)
            .dropna(subset=["_y", "Ano"])
            .groupby("Ano", observed=True)["_y"]
            .mean()
            .reset_index()
            .sort_values("Ano")
        )
        fig4 = px.line(ev, x="Ano", y="_y", markers=True, labels={"_y": f"Média {ind}", "Ano": "Ano"})
        fig4.update_traces(
            line=dict(color=_DASH_ACCENT, width=2.5),
            marker=dict(size=9, color=_DASH_ACCENT, line=dict(width=0)),
        )
        _dash_figure_layout(fig4, f"Média de {ind} por ano", calendar_year_x=True)
        st.plotly_chart(fig4, width="stretch", key=f"dbx_{ind}_evo")


def render_dashboards(df: pd.DataFrame, bundle: dict[str, Any]) -> None:
    st.title("Dashboards")
    st.caption(
        "Resumo dos dados carregados (Parquet). **Panorama** — visão geral; **Indicador** — um indicador de cada vez. "
        "Gráficos com fundo escuro alinhado ao tema da página."
    )

    with st.expander("O que significa cada indicador?", expanded=False):
        for code, title in INDICATOR_CATALOG:
            st.markdown(f"- **{code}** — {title}")
        _rl = int(RISKO_LIMIAR * 100)
        st.caption(
            f"«**Risco**» = probabilidade de alto risco (modelo ML). O cartão **«Alunos acima do corte»** diz **que % do grupo** "
            f"passa o patamar de **{_rl}%** de probabilidade — **não** é a média de risco nem «o risco ser {_rl}%»."
        )

    cdf = _ensure_risco(df, bundle)

    with st.container(border=True):
        st.subheader("Visão geral")
        _kpi_row(cdf)
        _rl = int(RISKO_LIMIAR * 100)
        st.caption(
            f"**Último cartão («Alunos acima do corte»):** **que % do grupo** o modelo classifica com probabilidade **≥ {_rl}%** "
            f"(o **{_rl}%** é só o patamar; o **valor do cartão** é a **fatia de alunos**). Ajuda no cartão para mais pormenor."
        )

    tab_pan, tab_ind = st.tabs(["Panorama", "Indicador"])

    with tab_pan:
        r1c1, r1c2 = st.columns(2, gap="small")
        with r1c1:
            if "IAN" in cdf.columns and "Ano" in cdf.columns:
                st.plotly_chart(_chart_ian_por_ano(cdf), width="stretch", key="dash_ian_ano")
            else:
                st.info("IAN ou Ano em falta.")
        with r1c2:
            if "Ano" in cdf.columns:
                st.plotly_chart(_chart_base_por_ano(cdf), width="stretch", key="dash_base_ano")
        r2c1, r2c2 = st.columns(2, gap="small")
        with r2c1:
            if "INDE" in cdf.columns and "Pedra" in cdf.columns:
                st.plotly_chart(_chart_inde_pedra(cdf), width="stretch", key="dash_inde_pedra")
        with r2c2:
            st.plotly_chart(_chart_correlacao(cdf), width="stretch", key="dash_corr")
        st.plotly_chart(_chart_risco_dist(cdf), width="stretch", key="dash_risco")

    with tab_ind:
        available = _numeric_indicators_in_df(cdf)
        if not available:
            st.warning("Nenhum indicador numérico encontrado.")
        else:
            labels = {c: f"{c} — {_indicator_label(c)}" for c in available}
            default_ix = available.index("IDA") if "IDA" in available else 0
            choice = st.selectbox(
                "Indicador",
                options=available,
                index=min(default_ix, len(available) - 1),
                format_func=lambda c: labels.get(c, c),
                key="pm_dash_indicator_select",
            )
            _charts_for_indicator(cdf, choice)

    with st.expander("Parecer do Theo sobre os dashboards (opcional)", expanded=False):
        st.caption(
            "Opcional: **não** afecta KPIs nem gráficos. Com **Ollama** em máquina local gera ou actualiza texto; "
            "no **Streamlit Cloud** (MVP sem LLM) verá a mensagem de indisponibilidade ou texto em cache se existir."
        )
        render_dashboard_theo_feedback(cdf, compact=True)
