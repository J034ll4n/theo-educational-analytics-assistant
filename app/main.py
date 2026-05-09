"""Entrada Streamlit — Theo, relatório anual, painel de risco e dicionário."""

from __future__ import annotations

import warnings

# LangChain importa `pydantic.v1` internamente; no Python 3.14+ o Pydantic emite este aviso até migração upstream.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

import base64
import hashlib
import html
import io
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Garante imports `app` e `passos_magico` quando o CWD não é a raiz do projeto
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _sidebar_slogan_data_uri(path: Path) -> str | None:
    """Data URI para embutir o slogan na hero da sidebar (controle total de layout/CSS)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.standard_b64encode(raw).decode('ascii')}"

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import plotly.io as pio
from plotly.io import to_image

from app.cached import cached_load_dados, cached_model_bundle, make_chat_sql_runner
from passos_magico.data_engine.loader import DATA_DIR, get_parquet_path
from passos_magico.llm.error_messages import humanize_sql_execution_error
from passos_magico.llm.ml_text import generate_diagnosis_text
from passos_magico.llm.ollama_client import ollama_available
from passos_magico.llm.institutional_router import is_institutional_narrative_only
from passos_magico.llm.pipeline import (
    invoke_institutional_insight,
    invoke_insight_text,
    sql_and_chart_step,
    stream_institutional_insight_text,
    stream_insight_text,
    suggestions_step,
)
from passos_magico.ml.features import (
    FEATURE_ORDER,
    latest_row_per_ra_table,
    one_row_per_ra_for_year,
    pick_latest_year_row,
    reference_years_available,
    row_features_from_df,
    rows_for_reference_year,
    years_for_ra,
)
from passos_magico.ml.inference import (
    explain_row_shap,
    predict_risk_slice,
    predict_row_after_simulation,
    predict_row_features,
)
from passos_magico.ml.risk_display import OPERATIONAL_HIGH_RISK_THRESHOLD
from passos_magico.llm.charts import CHART_TYPE_OPTIONS, figure_from_dataframe, heuristic_kind_to_chart_id
from passos_magico.semantic.dictionary_merge import merge_dictionary_with_dataframe
from passos_magico.semantic.metadata import (
    load_annual_summary_text,
    load_dictionary,
    load_gamma_context_text,
    merge_theo_context_blocks,
    rows_to_prompt_block,
    save_dictionary,
)
from passos_magico.ui.dashboard import render_dashboards
from passos_magico.ui.risk_sim_copy import risk_explain_lines, sim_matches_base_ficha, snapshot_sim_baseline
from passos_magico.ui.ranking_filters import (
    ranking_fase_options,
    ranking_mask,
    ranking_turma_letter_options_for_fase,
)
from passos_magico.ui.styles import inject_global_css
from tests.fixtures.theo_question_catalog import theo_test_question_groups

# Páginas: id interno, rótulo curto no menu, descrição para a lateral
PAGE_DEFS: list[tuple[str, str, str]] = [
    (
        "chat",
        "💬  Chat analítico",
        "Converse com o Theo: ele gera SQL nos seus dados (DuckDB), mostra um gráfico e explica o resultado. "
        "Use perguntas sobre médias, turmas, anos ou indicadores (IDA, INDE, etc.). Tudo roda localmente com Ollama.",
    ),
    (
        "annual_report",
        "📄  Relatório anual",
        "Relatório institucional (Gamma) embebido nesta página.",
    ),
    (
        "risk",
        "🎯  Previsão de risco",
        "Ficha individual com SHAP e parecer do Theo; **matriz de priorização** com filtros por fase (1–8) e turma (A–E), "
        "indicadores de recorte e exportação CSV. Simulação só no modo técnico (opcional).",
    ),
    (
        "dashboard",
        "📊  Dashboards",
        "KPIs agregados (alunos, INDE, IDA, adequação IAN, risco de defasagem) e gráficos sobre o Parquet: panorama e análise por indicador.",
    ),
    (
        "dict",
        "📖  Dicionário de dados",
        "Edite descrições das colunas: o Theo usa esse texto como contexto ao gerar SQL e respostas. "
        "Salve para aplicar na próxima pergunta do chat. Na mesma página pode pré-visualizar o **Parquet** em uso (e opcionalmente o CSV de entrada ao ETL).",
    ),
]
PAGE_LABELS: dict[str, str] = {p[0]: p[1] for p in PAGE_DEFS}
PAGE_HELP: dict[str, str] = {p[0]: p[2] for p in PAGE_DEFS}
_MAX_CHART_ROWS = 5000

_DEFAULT_ANNUAL_REPORT_URL = "https://datathon-passos-magicos-sojo7d1.gamma.site/"


def _annual_gamma_url() -> str:
    """URL público da apresentação no Gamma. Ambiente → secrets → URL por defeito."""
    env = os.environ.get("PM_RELATORIO_ANUAL_GAMMA_URL", "").strip()
    if env:
        return env
    try:
        v = st.secrets["RELATORIO_ANUAL_GAMMA_URL"]
        if v and str(v).strip():
            return str(v).strip()
    except Exception:
        pass
    return _DEFAULT_ANNUAL_REPORT_URL


def _annual_plain_for_theo() -> str:
    """Conteúdo de `resumo_anual.txt` para o contexto do Theo."""
    return load_annual_summary_text()

# Perguntas de exemplo = catálogo testado em CI (tests/test_sql_catalog.py)
THEO_TEST_QUESTIONS: tuple[tuple[str, list[str]], ...] = theo_test_question_groups(4)


def _serialize_df_for_chart(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    try:
        return df.head(_MAX_CHART_ROWS).to_json(orient="split", date_format="iso", default_handler=str)
    except Exception:
        return None


def _deserialize_chart_df(raw: str | None) -> pd.DataFrame | None:
    if not raw:
        return None
    try:
        return pd.read_json(io.StringIO(raw), orient="split")
    except Exception:
        return None


def _chart_type_labels() -> dict[str, str]:
    return {cid: label for cid, label in CHART_TYPE_OPTIONS}


def _scalar_result_no_chart(df: pd.DataFrame | None, heuristic_kind: str | None) -> bool:
    """Uma linha agregada sem eixo categórico — o Theo não desenha série útil (modo kpi nos gráficos)."""
    hk = (heuristic_kind or "").strip().lower()
    if hk == "kpi":
        return True
    # Mensagens antigas sem `chart_kind`: mesmo critério heurístico de `charts.py` (1 linha, só números).
    if not hk and df is not None and not df.empty and len(df) == 1:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cats = [c for c in df.columns if c not in nums]
        return len(nums) >= 1 and len(cats) == 0
    return False


def _render_chart_with_type_switch(
    *,
    df: pd.DataFrame | None,
    figure_json_fallback: str | None,
    msg_index: int,
    msg_id: int,
    default_heuristic_kind: str | None,
    download_key: str,
) -> None:
    """Mostra o gráfico com seletor de tipo; usa dados tabulares quando existirem."""
    labels = _chart_type_labels()
    ids = [cid for cid, _ in CHART_TYPE_OPTIONS]
    df_ok = df is not None and not df.empty
    if df_ok and _scalar_result_no_chart(df, default_heuristic_kind):
        st.caption(
            "Resultado **agregado** (um número ou poucos valores) — **sem gráfico**. "
            "A interpretação está na **Resposta** abaixo; abra **SQL gerado** para ver a consulta exata."
        )
        return
    if df_ok:
        default_id = heuristic_kind_to_chart_id(default_heuristic_kind or "auto")
        if default_id not in ids:
            default_id = "auto"
        pre_index = ids.index(default_id)
        key_sel = f"chart_kind_{msg_index}_{msg_id}"
        choice = st.selectbox(
            "Tipo de gráfico",
            options=ids,
            index=pre_index,
            format_func=lambda x: labels[x],
            key=key_sel,
            help="Se o gráfico automático ficou confuso, escolha outro tipo usando os mesmos dados da consulta.",
        )
        fig, _ = figure_from_dataframe(df, choice)
        st.plotly_chart(
            fig,
            width="stretch",
            key=f"theo_plot_df_{msg_index}_{msg_id}_{choice}",
        )
        try:
            img_bytes = to_image(fig, format="png", engine="kaleido")
            st.download_button(
                label="Baixar gráfico (PNG)",
                data=img_bytes,
                file_name="grafico_theo.png",
                mime="image/png",
                key=f"{download_key}_png",
            )
        except Exception:
            st.caption("Exportação PNG requer Kaleido instalado corretamente.")
        return

    if figure_json_fallback:
        fig = pio.from_json(figure_json_fallback)
        st.plotly_chart(
            fig,
            width="stretch",
            key=f"theo_plot_json_{msg_index}_{msg_id}",
        )
        try:
            img_bytes = to_image(fig, format="png", engine="kaleido")
            st.download_button(
                label="Baixar gráfico (PNG)",
                data=img_bytes,
                file_name="grafico_theo.png",
                mime="image/png",
                key=f"{download_key}_png",
            )
        except Exception:
            st.caption("Exportação PNG requer Kaleido instalado corretamente.")


def risk_color(p: float) -> str:
    if p < OPERATIONAL_HIGH_RISK_THRESHOLD:
        return "#3fb950"
    if p < 0.65:
        return "#d29922"
    return "#f85149"


def _risk_band_index(p: float) -> int:
    if p < OPERATIONAL_HIGH_RISK_THRESHOLD:
        return 0
    if p < 0.65:
        return 1
    return 2


def _risk_ficha_panel_rgba(proba: float) -> tuple[str, str]:
    """(fundo suave, borda) em rgba alinhados a `risk_color`."""
    idx = _risk_band_index(proba)
    if idx == 0:
        return "rgba(63, 185, 80, 0.12)", "rgba(63, 185, 80, 0.55)"
    if idx == 1:
        return "rgba(210, 153, 34, 0.14)", "rgba(210, 153, 34, 0.65)"
    return "rgba(248, 81, 73, 0.12)", "rgba(248, 81, 73, 0.65)"


def _risk_zones_html(proba: float) -> str:
    """Três faixas com destaque na faixa ativa."""
    idx = _risk_band_index(proba)
    zones = [
        ("Baixa", "até 46%", "#3fb950"),
        ("Moderada", "46% a 65%", "#d29922"),
        ("Elevada", "acima de 65%", "#f85149"),
    ]
    parts = ['<div class="pm-risk-zones">']
    for i, (title, sub, col) in enumerate(zones):
        active = " pm-active" if i == idx else ""
        if i == idx:
            stl = f"border-color: {col}; background: {col}22; box-shadow: 0 0 18px {col}55;"
        else:
            stl = f"border-color: {col}44; background: {col}0a;"
        parts.append(
            f'<div class="pm-risk-zone{active}" style="{stl}">'
            f"<span>{title}</span>{sub}</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _shap_risk_figure(shap_pairs: list[tuple[str, float]]) -> go.Figure:
    """Barras horizontais: verde = empurra risco para baixo, vermelho = aumenta."""
    labels = [p[0] for p in shap_pairs[:12]]
    vals = [p[1] for p in shap_pairs[:12]]
    colors: list[str] = []
    for v in vals:
        if v < 0:
            colors.append("rgba(63, 185, 80, 0.92)")
        elif v > 0:
            colors.append("rgba(248, 81, 73, 0.88)")
        else:
            colors.append("rgba(139, 148, 158, 0.7)")
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.35)", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="O que mais empurrou o risco neste perfil",
            font=dict(size=15),
        ),
        margin=dict(l=130, r=24, t=56, b=48),
        height=min(540, max(300, 28 * len(labels) + 100)),
        xaxis=dict(
            title="→ direita = mais risco neste modelo · ← esquerda = menos risco",
            title_font=dict(size=11),
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.25)",
        ),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(13,17,23,0.4)",
        plot_bgcolor="rgba(13,17,23,0.55)",
    )
    return fig


def _suggestions_cache_key(user_q: str, insight: str) -> str:
    payload = f"{user_q[:400]}\n{insight[:900]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:28]


def _cached_suggestions(user_text: str, insight_full: str) -> list[str]:
    cache: dict[str, list[str]] = st.session_state.setdefault("pm_theo_suggestions_cache", {})
    key = _suggestions_cache_key(user_text, insight_full)
    if key in cache:
        return cache[key]
    out = suggestions_step(user_text, insight_full)
    cache[key] = out
    while len(cache) > 64:
        cache.pop(next(iter(cache)))
    return out


def _render_one_chat_message(msg: dict, msg_index: int) -> None:
    av = "🎓" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=av):
        if msg["role"] == "assistant":
            if msg.get("sql"):
                with st.expander("SQL gerado"):
                    st.code(msg["sql"], language="sql")
            if msg.get("figure_json") or msg.get("chart_df_json"):
                d_hist = _deserialize_chart_df(msg.get("chart_df_json"))
                _render_chart_with_type_switch(
                    df=d_hist,
                    figure_json_fallback=msg.get("figure_json"),
                    msg_index=msg_index,
                    msg_id=int(msg.get("id", 0)),
                    default_heuristic_kind=msg.get("chart_kind"),
                    download_key=f"dl_{msg_index}_{msg.get('id', 0)}",
                )
            st.markdown(msg.get("content", ""))
            raw_err = msg.get("sql_error_raw")
            if raw_err:
                with st.expander("Detalhe técnico (DuckDB / validação)"):
                    st.code(str(raw_err), language="text")
            if msg.get("suggestions"):
                cols = st.columns(3)
                for i, s in enumerate(msg["suggestions"][:3]):
                    if cols[i].button(s, key=f"sg_{msg_index}_{msg.get('id', 0)}_{i}"):
                        st.session_state.pending_prompt = s
                        st.rerun()
        else:
            st.markdown(msg["content"])


_GITHUB_CHATBOT_REPO = "https://github.com/J034ll4n/theo-educational-analytics-assistant"


def render_chat(
    theo_context_block: str,
    ollama_ok: bool,
    sql_runner,
    sql_context_df: pd.DataFrame | None = None,
) -> None:
    pending = st.session_state.pop("pending_prompt", None)

    st.subheader("Chat analítico — Theo")
    if not ollama_ok:
        st.info(
            "**Ollama desligado ou indisponível** neste ambiente (por exemplo no Streamlit Cloud). "
            "Para usar o **chatbot** completo do Theo, descarregue o projeto no GitHub, instale o **Ollama** no seu PC "
            "e execute a aplicação **localmente**."
        )
        st.markdown(f"[Abrir repositório no GitHub — download e instruções]({_GITHUB_CHATBOT_REPO})")
    st.caption(
        "O Theo gera SQL, **gráfico quando faz sentido** (totais de uma linha: só texto, sem gráfico nem tabela duplicada) e explica. "
        "Quando há figura, pode mudar o tipo de gráfico sem repetir a pergunta."
    )
    with st.expander("Exemplos de perguntas para testar o Theo", expanded=False):
        for title, items in THEO_TEST_QUESTIONS:
            st.markdown(f"**{title}**")
            for i, q in enumerate(items, start=1):
                st.markdown(f"{i}. {q}")

    # Histórico primeiro; chat_input por último para a barra ficar no rodapé e a área principal rolar.
    for idx, msg in enumerate(st.session_state.messages):
        _render_one_chat_message(msg, idx)

    prompt = st.chat_input(
        "Faça uma pergunta sobre os dados educacionais…",
        key="pm_chat_input",
    )
    user_text = (pending or (prompt or "")).strip()
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.rerun()

    msgs = st.session_state.messages
    if not msgs or msgs[-1]["role"] != "user":
        return

    user_text = msgs[-1]["content"]
    msg_id = len(msgs)

    with st.chat_message("assistant", avatar="🎓"):
        try:
            _run_assistant_turn(
                user_text,
                theo_context_block,
                ollama_ok,
                sql_runner,
                msg_id,
                sql_context_df=sql_context_df,
            )
        except Exception as e:
            err_msg = (
                "**Theo:** Algo falhou por aqui do lado da aplicação — não foi o seu raciocínio. "
                "Tente outra vez; se insistir, copie o detalhe técnico para o apoio."
            )
            st.markdown(err_msg)
            with st.expander("Detalhe técnico"):
                st.code(str(e), language="text")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "id": msg_id,
                    "content": err_msg,
                    "sql": None,
                    "sql_error_raw": str(e),
                    "figure_json": None,
                    "suggestions": [
                        "Tentar uma pergunta mais simples",
                        "Média de IDA por Ano",
                        "Listar turmas com mais alunos",
                    ],
                }
            )


def _run_assistant_turn(
    user_text: str,
    theo_context_block: str,
    ollama_ok: bool,
    sql_runner,
    msg_id: int,
    sql_context_df: pd.DataFrame | None = None,
) -> None:
    # Perguntas só sobre o texto do resumo anual — não gerar SQL em `dados`
    if ollama_ok and is_institutional_narrative_only(user_text, theo_context_block):
        with st.spinner("Lendo o resumo institucional…"):
            insight_full = st.write_stream(
                stream_institutional_insight_text(user_text, theo_context_block)
            )
        if insight_full is None:
            insight_full = ""
        if len(insight_full.strip()) < 40:
            insight_full = invoke_institutional_insight(user_text, theo_context_block)
        sugs = _cached_suggestions(user_text, insight_full)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "id": msg_id,
                "content": insight_full,
                "sql": "-- Pergunta respondida com o texto institucional (sem consulta à base tabular).",
                "figure_json": None,
                "chart_df_json": None,
                "chart_kind": "institucional",
                "suggestions": sugs,
            }
        )
        return

    with st.spinner("Buscando dados…"):
        step = sql_and_chart_step(
            user_text,
            theo_context_block,
            ollama_ok,
            sql_executor=sql_runner,
            sql_context_df=sql_context_df,
        )
    if step.sql_error or step.df is None:
        err = step.sql_error or "Erro desconhecido."
        if ollama_ok and getattr(step, "recovery_markdown", None):
            text = f"**Theo:**\n\n{step.recovery_markdown}"
            raw_for_msg: str | None = None
        elif ollama_ok:
            friendly = humanize_sql_execution_error(str(err))
            text = f"**Theo:** {friendly}"
            raw_for_msg = str(err)
        else:
            text = "**Theo:** O **Ollama** não está disponível neste ambiente. Inicie o serviço local e tente de novo."
            raw_for_msg = None
        st.markdown(text)
        if raw_for_msg:
            with st.expander("Detalhe técnico (DuckDB / validação)"):
                st.code(raw_for_msg, language="text")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "id": msg_id,
                "content": text,
                "sql": step.sql,
                "sql_error_raw": raw_for_msg,
                "figure_json": None,
                "suggestions": [
                    "Média de IDA por Fase em 2022",
                    "Alunos por Turma no último ano",
                    "Evolução do INDE por Ano",
                ],
            }
        )
        return

    assert step.figure is not None
    if step.sql:
        with st.expander("SQL gerado"):
            st.code(step.sql, language="sql")
    _render_chart_with_type_switch(
        df=step.df,
        figure_json_fallback=step.figure.to_json(),
        msg_index=msg_id,
        msg_id=msg_id,
        default_heuristic_kind=step.chart_kind,
        download_key=f"dl_live_{msg_id}",
    )

    with st.spinner("Pensando…"):
        insight_full = st.write_stream(
            stream_insight_text(user_text, theo_context_block, step.df, step.chart_kind)
        )
    if insight_full is None:
        insight_full = ""
    if len(insight_full.strip()) < 40:
        insight_full = invoke_insight_text(
            user_text, theo_context_block, step.df, step.chart_kind
        )
    sugs = _cached_suggestions(user_text, insight_full)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "id": msg_id,
            "content": insight_full,
            "sql": step.sql,
            "figure_json": step.figure.to_json(),
            "chart_df_json": _serialize_df_for_chart(step.df),
            "chart_kind": step.chart_kind,
            "suggestions": sugs,
        }
    )


def _turma_ord_to_letter(ordv: float) -> str:
    return {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}.get(int(round(float(ordv))), "—")


def _pedra_ord_to_name(ordv: float) -> str:
    return {1: "Quartzo", 2: "Ágata", 3: "Ametista", 4: "Topázio"}.get(int(round(float(ordv))), "—")


def _risk_heading_html(title: str, subtitle: str | None = None, *, tight_top: bool = False) -> str:
    """Título de seção + subtítulo opcional (página Previsão de risco)."""
    extra = " pm-risk-section-tight-top" if tight_top else ""
    sub = (
        f'<p class="pm-risk-section-sub">{subtitle}</p>'
        if subtitle
        else ""
    )
    return f'<p class="pm-risk-section-title{extra}">{title}</p>{sub}'


def _risk_subheading_muted(title: str, subtitle: str | None = None) -> str:
    """Subseção sem barra inferior — coluna direita do simulador e blocos densos."""
    sub = (
        f'<p class="pm-risk-section-sub-muted">{subtitle}</p>'
        if subtitle
        else ""
    )
    return f'<p class="pm-risk-section-title-muted">{title}</p>{sub}'


def _eng_float(row: pd.Series | None, key: str, default: float = 0.0) -> float:
    if row is None or key not in row.index:
        return default
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _media_turma_base_optional(row: pd.Series | None) -> float | None:
    if row is None or "media_turma_inde" not in row.index:
        return None
    v = row.get("media_turma_inde")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sim_ficha_controle_resumo_html(
    baseline: dict[str, float],
    key: str,
    current: float,
    *,
    tol: float = 0.051,
    nd: int = 1,
) -> str:
    """Resumo legível: valor na ficha, valor no controle e mudança % (verde = acima da ficha, vermelho = abaixo)."""
    bv = float(baseline.get(key, current))
    cv = float(current)
    d = cv - bv
    if abs(d) <= tol:
        mud = '<span style="color:#6e7681;font-weight:600;">sem mudança vs ficha</span>'
    else:
        den = max(abs(bv), 0.45)
        if key == "Delta_INDE":
            den = max(abs(bv), 0.35) if abs(bv) > 1e-6 else 1.0
        pct = (d / den) * 100.0
        pct = float(max(-500.0, min(500.0, pct)))
        if pct > 0.5:
            col, lab = "#3fb950", f"+{pct:.0f}% vs ficha (subiu no controle)"
        elif pct < -0.5:
            col, lab = "#f85149", f"{pct:.0f}% vs ficha (desceu no controle)"
        else:
            col, lab = "#8b949e", f"{pct:+.0f}% vs ficha"
        mud = f'<span style="color:{col};font-weight:700;">{lab}</span>'
    return (
        f'<p style="margin:0.2rem 0 0.65rem 0;font-size:0.88rem;line-height:1.4;">'
        f'<span style="color:#8b949e">Na ficha:</span> <strong>{bv:.{nd}f}</strong>'
        f' &nbsp;·&nbsp; <span style="color:#8b949e">No controle:</span> <strong>{cv:.{nd}f}</strong>'
        f" &nbsp;·&nbsp; {mud}</p>"
    )


def _get_engineered_row_for_display(
    df: pd.DataFrame, ra: str, ref_year: int | None = None
) -> pd.Series | None:
    """Linha alinhada a `row_features_from_df` (último ano ou `ref_year`), com engenharia do modelo de risco."""
    from passos_magico.ml.features import single_row_for_ra_and_year
    from passos_magico.ml.risk_pipeline import ensure_risk_engineering

    ra_col = "RA" if "RA" in df.columns else "ra"
    if ra_col not in df.columns:
        return None
    sub = df[df[ra_col].astype(str) == str(ra)]
    if sub.empty:
        return None
    if ref_year is None:
        key = pick_latest_year_row(sub).iloc[0]
    else:
        one = single_row_for_ra_and_year(df, ra, int(ref_year))
        if one.empty:
            return None
        key = one.iloc[0]
    y_val: float | None = None
    for ac in ("ano_referencia", "Ano"):
        if ac in key.index and pd.notna(key[ac]):
            try:
                y_val = float(key[ac])
            except (TypeError, ValueError):
                y_val = None
            break
    eng = ensure_risk_engineering(df)
    if ra_col not in eng.columns:
        return None
    cand = eng[eng[ra_col].astype(str) == str(ra)].copy()
    if y_val is not None:
        for ac in ("ano_referencia", "Ano"):
            if ac in cand.columns:
                s = pd.to_numeric(cand[ac], errors="coerce")
                cand = cand[s.sub(y_val).abs() < 0.51]
                break
    if cand.empty:
        return None
    return cand.iloc[0]


def _fmt_pedagogy_num(val: Any, nd: int = 2) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        return f"{float(val):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _render_pedagogy_context_trajectory(row: pd.Series | None) -> None:
    """Cartões em linguagem de equipe: distância à média da turma e trajetória do INDE."""
    if row is None:
        return

    inde = row.get("inde") if "inde" in row.index else row.get("INDE")
    dist = row.get("distancia_media_turma")
    media_t = row.get("media_turma_inde")
    delta_inde = row.get("delta_inde")
    tend = row.get("tendencia_inde")

    has_real_group = media_t is not None and not pd.isna(media_t)
    inde_ok = inde is not None and not pd.isna(inde)
    if not has_real_group or not inde_ok:
        metric_line = (
            f"<p class=\"pm-risk-pedagogy-metric\">INDE do aluno: <strong>{_fmt_pedagogy_num(inde)}</strong></p>"
            if inde_ok
            else ""
        )
        dist_txt = (
            "Não foi possível comparar com a **média do grupo** neste registro "
            "(falta combinação instituição/fase/ano na base ou INDE ausente)."
        )
        dist_detail = ""
    elif dist is None or (isinstance(dist, float) and pd.isna(dist)):
        metric_line = (
            f"<p class=\"pm-risk-pedagogy-metric\">INDE do aluno: <strong>{_fmt_pedagogy_num(inde)}</strong> · "
            f"Média INDE do grupo: <strong>{_fmt_pedagogy_num(media_t)}</strong></p>"
        )
        dist_txt = "Indicador de distância não disponível."
        dist_detail = ""
    else:
        d = float(dist)
        metric_line = (
            f"<p class=\"pm-risk-pedagogy-metric\">INDE do aluno: <strong>{_fmt_pedagogy_num(inde)}</strong> · "
            f"Média INDE do grupo: <strong>{_fmt_pedagogy_num(media_t)}</strong> · "
            f"Diferença (aluno − média): <strong>{_fmt_pedagogy_num(dist)}</strong></p>"
        )
        dist_txt = (
            f"O INDE deste aluno está <strong>{_fmt_pedagogy_num(abs(d))}</strong> ponto(s) "
            f"{'acima' if d >= 0 else 'abaixo'} da média INDE dos colegas no mesmo contexto "
            f"(mesma instituição, fase e ano de referência na base)."
        )
        if d <= -1.0:
            dist_detail = (
                " Uma distância negativa maior sugere **desalinhamento forte** em relação ao grupo — "
                "vale conversar sobre reforço, acompanhamento ou, em casos extremos, "
                "rever se a turma/nível é o mais adequado."
            )
        elif d < -0.3:
            dist_detail = (
                " Há espaço para **monitoria** ou planos curtos para aproximar o desempenho "
                "relativo dos colegas da mesma turma."
            )
        elif d <= 0.3:
            dist_detail = " Está **próximo da média** do grupo — o foco pode ser mais qualitativo (motivação, frequência, etc.)."
        else:
            dist_detail = " Está **acima da média** do grupo neste recorte — manter o que funciona e cuidar do bem-estar."

    if delta_inde is None or (isinstance(delta_inde, float) and pd.isna(delta_inde)):
        traj_txt = "Não foi possível calcular a variação do INDE."
    else:
        di = float(delta_inde)
        if abs(di) < 1e-6:
            traj_txt = (
                "Não há **variação** em relação ao registro anterior deste aluno na base "
                "(primeiro ano disponível ou mesmo valor)."
            )
        elif di > 0:
            traj_txt = (
                f"O INDE **subiu** {_fmt_pedagogy_num(di)} ponto(s) em relação ao ano anterior "
                "registado para este aluno — trajetória favorável."
            )
        else:
            traj_txt = (
                f"O INDE **desceu** {_fmt_pedagogy_num(abs(di))} ponto(s) em relação ao ano anterior "
                "— o modelo costuma ser sensível a quedas, mesmo pequenas."
            )

    tend_txt = ""
    if tend is not None and not (isinstance(tend, float) and pd.isna(tend)):
        t = float(tend)
        if abs(t) >= 0.05:
            tend_txt = (
                f"<p class=\"pm-risk-pedagogy-metric\">Tendência recente (média das mudanças de INDE): "
                f"<strong>{_fmt_pedagogy_num(t)}</strong> /ano (aprox.).</p>"
            )
            if t < -0.05:
                tend_txt += (
                    "<p>Vale planejar o **próximo período** com metas claras para recuperar o INDE "
                    "(ex.: ações que aumentem aprendizagem e engajamento de forma sustentada).</p>"
                )
            else:
                tend_txt += "<p>A tendência recente aponta para **estabilidade ou recuperação**.</p>"

    pc1, pc2 = st.columns(2, gap="medium")
    with pc1:
        st.markdown("**Contexto — aluno e turma**")
        st.markdown(
            f"{metric_line}<p>{dist_txt}{dist_detail}</p>",
            unsafe_allow_html=True,
        )
    with pc2:
        st.markdown("**Trajetória — INDE no tempo**")
        st.markdown(f"<p>{traj_txt}</p>{tend_txt}", unsafe_allow_html=True)
    st.caption("Texto a partir do relatório. A decisão é sempre da equipe; use o SHAP abaixo para prioridades.")


def render_risk(df: pd.DataFrame, bundle: dict[str, Any], theo_context_block: str) -> None:
    st.subheader("Previsão de risco escolar")
    st.markdown(
        "Probabilidade de **alto risco** (defasagem / evasão) a partir do relatório. "
        "Não substitui a equipa pedagógica. **Aba 1:** ficha individual (contexto, SHAP, Theo). **Aba 2:** matriz de priorização "
        "com filtros e exportação. **Simulação** só no modo técnico."
    )
    with st.expander("Ajuda rápida (SHAP, 46%, simulação)", expanded=False):
        st.markdown(
            "- **Risco:** estimativa do modelo sobre a ficha; **≤46%** baixo, **46–65%** atenção, **>65%** mais acompanhamento.\n"
            "- **SHAP:** barra à **direita** da linha = aumenta o risco neste modelo; à **esquerda** = reduz.\n"
            "- **Theo:** usa risco da ficha + SHAP; não inventa percentagens.\n"
            "- **Modo técnico:** cenário «e se…»; a base **não** é alterada."
        )

    tab1, tab2 = st.tabs(["Análise individual", "Matriz de priorização"])

    with tab1:
        st.markdown(
            _risk_heading_html(
                "1 · Quem analisar",
                "Uma só lista: **escreve** para filtrar e escolhe o aluno; se houver vários anos, escolhe o **ano da ficha** abaixo (por defeito 2024 se existir).",
                tight_top=True,
            ),
            unsafe_allow_html=True,
        )
        _picker_df = latest_row_per_ra_table(df)
        _nome_col = "Nome" if "Nome" in _picker_df.columns else "nome"
        _ra_col_p = "RA" if "RA" in _picker_df.columns else "ra"
        _n_alunos = len(_picker_df)
        _label_series = _picker_df[_nome_col].astype(str) + " — " + _picker_df[_ra_col_p].astype(str)
        ra_from_select = None
        if _n_alunos == 0:
            st.warning("Não há alunos na base carregada.")
            pick_display = None
        else:
            st.caption(
                f"**{_n_alunos:,}** aluno(s) na lista (um nome por aluno — **último ano** na base para o filtro). O campo abaixo filtra **enquanto digitas**."
                .replace(",", ".")
            )
            pick_display = st.selectbox(
                "Escolher aluno na lista",
                options=list(_label_series),
                index=None,
                placeholder="Digite nome, parte do RA ou cole o RA completo…",
                filter_mode="contains",
                help="Abre a lista, escreve para afinar resultados e clica no aluno. O filtro é na própria caixa (Streamlit).",
                key="pm_rsk_student_pick",
                width="stretch",
            )
        if pick_display:
            ra_from_select = str(_picker_df.loc[_label_series == pick_display, _ra_col_p].iloc[0])
        ra = (ra_from_select or "").strip()
        feats = None
        if ra:
            _years_ra = years_for_ra(df, ra)
            if not _years_ra:
                st.warning("Este RA não tem **Ano** / **ano_referencia** na base — não dá para montar a ficha por ano.")
            elif len(_years_ra) == 1:
                feats = row_features_from_df(df, ra, int(_years_ra[0]))
            else:
                _def_ficha_y = 2024 if 2024 in _years_ra else int(_years_ra[0])
                _y_idx = _years_ra.index(_def_ficha_y)
                _sel_ficha_y = st.selectbox(
                    "Ano da ficha",
                    options=_years_ra,
                    index=_y_idx,
                    format_func=lambda y: str(int(y)),
                    key=f"pm_rsk_ficha_year_{ra}",
                    help="Risco, SHAP e Theo usam os dados **deste** ano. Por defeito: **2024** se existir, senão o mais recente.",
                )
                feats = row_features_from_df(df, ra, int(_sel_ficha_y))

        if feats:
            base_feats = {k: feats[k] for k in FEATURE_ORDER}
            proba = predict_row_features(bundle, feats, df=df)
            shap_pairs = explain_row_shap(bundle, feats, df=df)
            nome = str(feats.get("Nome", ""))
            bidx = _risk_band_index(proba)
            band_names = ("Faixa baixa", "Faixa moderada", "Faixa elevada")
            band_hints = (
                "Menor probabilidade de alto risco segundo o modelo — manter acompanhamento habitual.",
                "Zona de atenção — vale cruzar com indicadores e rede de apoio.",
                "Maior probabilidade no modelo — priorizar olhar próximo e planejamento pedagógico.",
            )

            st.markdown(_risk_heading_html("2 · Risco na ficha (dados da base)"), unsafe_allow_html=True)
            meta1, meta2, meta3, meta4 = st.columns(4)
            meta1.metric("Aluno", nome or "—")
            meta2.metric("Fase", f"{int(feats['Fase'])}")
            meta3.metric("Ano", f"{int(feats['Ano'])}")
            meta4.metric(
                "Turma",
                f"{_turma_ord_to_letter(feats['Turma_ord'])}",
                help=f"Na base: {int(feats['Turma_ord'])} (1=A … 5=E).",
            )
            st.caption(
                f"Risco e SHAP usam o registo de **{int(feats['Ano'])}**. Na matriz, escolhe o **mesmo ano** em «Ano de referência na lista» "
                "e **um registo por aluno** para alinhar a percentagem."
            )
            _rc = risk_color(proba)
            _bg, _bd = _risk_ficha_panel_rgba(proba)
            st.markdown(
                f'<div class="pm-risk-ficha-card" style="border:2px solid {_bd};background:{_bg};'
                f'border-radius:12px;padding:1rem 1.2rem;margin:0.6rem 0 0.85rem 0;">'
                f'<div style="font-size:0.8rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;'
                f'color:#8b949e;margin-bottom:0.35rem;">Probabilidade de alto risco (modelo)</div>'
                f'<div style="font-size:clamp(2rem,5vw,2.75rem);font-weight:800;line-height:1.1;color:{_rc};">'
                f"{proba * 100:.1f}%</div>"
                f'<div style="display:inline-block;margin-top:0.5rem;padding:0.25rem 0.65rem;border-radius:999px;'
                f'font-size:0.85rem;font-weight:700;border:1px solid {_rc};color:{_rc};background:rgba(13,17,23,0.35);">'
                f"{band_names[bidx]}</div>"
                f'<p style="margin:0.65rem 0 0 0;font-size:0.92rem;line-height:1.45;color:#c9d1d9;">{band_hints[bidx]}</p>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.caption("**Legenda de cores:** verde = baixo risco · âmbar = atenção · vermelho = risco elevado (sempre em probabilidade do modelo, não diagnóstico).")
            st.markdown(_risk_zones_html(proba), unsafe_allow_html=True)
            with st.expander("Nota sobre os limiares 46% e 65%", expanded=False):
                st.markdown(
                    "O modelo foi calibrado com **46%** como limiar operacional de «alto risco» na UI; **65%** marca uma "
                    "zona ainda mais alta na legenda. Valores **abaixo de 46%** aparecem em **verde** (faixa baixa)."
                )

            _eng_row = _get_engineered_row_for_display(df, ra, int(feats["Ano"]))
            st.markdown(
                _risk_heading_html(
                    "Olhar pedagógico",
                    "Posição em relação à média do grupo e evolução do INDE — texto gerado a partir do relatório.",
                ),
                unsafe_allow_html=True,
            )
            _render_pedagogy_context_trajectory(_eng_row)

            st.markdown(
                _risk_heading_html(
                    "3 · Fatores que mais influenciaram esta estimativa (SHAP)",
                    "Uma barra por variável neste aluno; compare com a linha tracejada no meio.",
                ),
                unsafe_allow_html=True,
            )
            fig_shap = _shap_risk_figure(shap_pairs)
            st.plotly_chart(fig_shap, width="stretch", key="pm_shap_chart")

            st.markdown(
                _risk_heading_html(
                    "4 · Parecer e orientação (Theo)",
                    "Leitura em linguagem natural com base no **risco da ficha**, no gráfico SHAP e no contexto carregado. "
                    "O Theo **não** inventa novas percentagens de risco nem substitui a equipe.",
                ),
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                _dict_ml = rows_to_prompt_block(merge_dictionary_with_dataframe(df, load_dictionary()))
                diag = generate_diagnosis_text(
                    nome,
                    ra,
                    proba,
                    shap_pairs,
                    theo_context_block=_dict_ml,
                    feats=feats,
                )
                st.markdown(diag)

            with st.expander("Modo técnico: simulação manual (opcional)", expanded=False):
                st.markdown(
                    "**Aviso:** a simulação altera só os indicadores dos **controles**; outras variáveis do modelo podem "
                    "permanecer alinhadas à **última ficha** na base — o resultado é **exploratório**, não substitui "
                    "o risco oficial da secção 2. A base de dados **não** é alterada."
                )
                _sim_active = st.checkbox(
                    "Ativar simulador (mostra os controles e recalcula a probabilidade)",
                    value=False,
                    key=f"pm_rsk_sim_on_{ra}_{int(feats['Ano'])}",
                )
                if not _sim_active:
                    st.caption(
                        "Ligue acima para abrir os **controles** e testar cenários em cima da ficha (sem gravar na base)."
                    )
                else:
                    st.markdown(
                        _risk_heading_html(
                            "Simulador «e se…?»",
                            "Deslize para testar outro cenário. **À direita** aparece o novo risco. Fase, turma, ano e pedra **seguem a ficha** (não dá para mudar aqui).",
                            tight_top=True,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Cada alteração recalcula o modelo (IAA−IDA, distância à média da turma, Δ INDE). A base **não** é alterada."
                    )

                    sim = dict(base_feats)
                    sim["IAA"] = _eng_float(_eng_row, "iaa", 5.0)
                    sim["IPS"] = _eng_float(_eng_row, "ips", 5.0)
                    sim["MAT"] = _eng_float(_eng_row, "mat", 5.0)
                    sim["POR"] = _eng_float(_eng_row, "por", 5.0)
                    sim["Delta_INDE"] = _eng_float(_eng_row, "delta_inde", 0.0)
        
                    _sim_baseline = snapshot_sim_baseline(sim)
        
                    st.info(
                        "**Dica:** em cada bloco, o deslizador está na **faixa numérica** (ex.: 0 a 10). "
                        "**Na ficha** = valor do relatório; **No controle** = valor que você está simulando agora. "
                        "A porcentagem colorida só compara esses dois (verde = controle **maior** que a ficha)."
                    )
        
                    col_ctrl, col_main = st.columns([0.42, 0.58], gap="medium")
        
                    with col_ctrl:
                        st.markdown("##### Atitude (impacto alto)")
                        st.markdown("**IEG — Engajamento** · escala 0 a 10")
                        sim["IEG"] = st.slider(
                            " ",
                            0.0,
                            10.0,
                            float(sim["IEG"]),
                            0.1,
                            key="pm_rsk_ieg",
                            label_visibility="collapsed",
                            help="Arraste para mudar o engajamento em relação ao valor da ficha.",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(_sim_baseline, "IEG", sim["IEG"]),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**IAA — Autoavaliação** · escala 0 a 10")
                        sim["IAA"] = st.slider(
                            " ",
                            0.0,
                            10.0,
                            float(sim["IAA"]),
                            0.1,
                            key="pm_rsk_iaa",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(_sim_baseline, "IAA", sim["IAA"]),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**IPS — Psicossocial** · escala 0 a 10")
                        sim["IPS"] = st.slider(
                            " ",
                            0.0,
                            10.0,
                            float(sim["IPS"]),
                            0.1,
                            key="pm_rsk_ips",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(_sim_baseline, "IPS", sim["IPS"]),
                            unsafe_allow_html=True,
                        )
        
                        st.markdown("##### Acadêmico")
                        st.markdown("**INDE** · escala 0 a 10")
                        sim["INDE"] = st.slider(
                            " ",
                            0.0,
                            10.0,
                            float(sim["INDE"]),
                            0.1,
                            key="pm_rsk_inde",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(_sim_baseline, "INDE", sim["INDE"]),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**IDA — Aprendizagem** · escala 0 a 10")
                        sim["IDA"] = st.slider(
                            " ",
                            0.0,
                            10.0,
                            float(sim["IDA"]),
                            0.1,
                            key="pm_rsk_ida",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(_sim_baseline, "IDA", sim["IDA"]),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**IAN** · escala 0 a 10")
                        sim["IAN"] = st.slider(
                            " ",
                            0.0,
                            10.0,
                            float(sim["IAN"]),
                            0.1,
                            key="pm_rsk_ian",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(_sim_baseline, "IAN", sim["IAN"]),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**MAT** · escala 0 a 10")
                        sim["MAT"] = st.slider(
                            " ",
                            0.0,
                            10.0,
                            float(sim["MAT"]),
                            0.1,
                            key="pm_rsk_mat",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(_sim_baseline, "MAT", sim["MAT"]),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**POR** · escala 0 a 10")
                        sim["POR"] = st.slider(
                            " ",
                            0.0,
                            10.0,
                            float(sim["POR"]),
                            0.1,
                            key="pm_rsk_por",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(_sim_baseline, "POR", sim["POR"]),
                            unsafe_allow_html=True,
                        )
        
                        st.markdown("##### Trajetória")
                        st.markdown("**Δ INDE esperado** · escala −5 a +5")
                        sim["Delta_INDE"] = st.slider(
                            " ",
                            -5.0,
                            5.0,
                            float(sim["Delta_INDE"]),
                            0.1,
                            key="pm_rsk_delta_inde",
                            label_visibility="collapsed",
                            help="Variação de INDE usada pelo modelo neste cenário.",
                        )
                        st.markdown(
                            _sim_ficha_controle_resumo_html(
                                _sim_baseline, "Delta_INDE", sim["Delta_INDE"], nd=2
                            ),
                            unsafe_allow_html=True,
                        )
                        _choque_ui = float(sim["IAA"]) - float(sim["IDA"])
                        st.caption(
                            f"**IAA − IDA** neste cenário: **{_choque_ui:+.1f}**. "
                            f"**Contexto fixo (só ficha, sem deslizador):** Fase **{int(base_feats['Fase'])}** · "
                            f"Turma **{_turma_ord_to_letter(base_feats['Turma_ord'])}** · Ano **{int(base_feats['Ano'])}** · "
                            f"Pedra **{_pedra_ord_to_name(base_feats['Pedra_ord'])}** · IPV **{float(sim['IPV']):.1f}**."
                        )
        
                    new_p = predict_row_after_simulation(bundle, df, ra, sim)
                    # Com os mesmos números da ficha, o caminho do simulador (reengenharia + derivadas) pode divergir
                    # da entrada usada na secção 2 (`row_matrix_for_ficha_feats`). Forçamos o mesmo risco da ficha.
                    if (
                        new_p is not None
                        and not (isinstance(new_p, float) and np.isnan(new_p))
                        and sim_matches_base_ficha(sim, _sim_baseline)
                    ):
                        new_p = float(proba)
        
                    with col_main:
                        if new_p is None or (isinstance(new_p, float) and np.isnan(new_p)):
                            st.error("Não foi possível calcular o risco simulado para este RA.")
                        else:
                            delta_pp = (new_p - proba) * 100.0
                            _sim_pct = float(new_p) * 100.0
                            if abs(delta_pp) < 0.08:
                                _sum_line = (
                                    "**Quase igual** à ficha original — os números batem com o registo na base."
                                    if sim_matches_base_ficha(sim, _sim_baseline)
                                    else "**Pouca diferença** em relação à ficha: mexer nos controles quase não mudou o resultado."
                                )
                            elif delta_pp > 0:
                                _sum_line = (
                                    f"**Sobe cerca de {delta_pp:.1f} pontos** na escala de 0 a 100 em relação à ficha — "
                                    "a estimativa fica **mais alta** do que no registo original."
                                )
                            else:
                                _sum_line = (
                                    f"**Desce cerca de {abs(delta_pp):.1f} pontos** na escala de 0 a 100 em relação à ficha — "
                                    "a estimativa fica **mais baixa** do que no registo original (**só** vale o que os números "
                                    "dos controles representam de verdade para o aluno)."
                                )

                            with st.container(border=True):
                                st.markdown(
                                    _risk_subheading_muted(
                                        "Interpretação rápida",
                                        "Tudo num só lugar: **ficha** = dados originais; **cenário** = o que você mudou nos deslizadores. "
                                        "O **gráfico de fatores** acima mostra o que mais puxa o resultado neste aluno.",
                                    ),
                                    unsafe_allow_html=True,
                                )

                                crit46 = float(new_p) > OPERATIONAL_HIGH_RISK_THRESHOLD
                                if crit46:
                                    st.warning(
                                        "**Zona de atenção:** neste cenário o modelo estima **mais de 46%** de probabilidade "
                                        "de alto risco (na escala deste painel). Cruze com o que a escola observa e com o gráfico de fatores."
                                    )
                                else:
                                    st.success(
                                        "**Zona verde:** neste cenário o modelo fica **até 46%** nessa escala. "
                                        "É um apoio à conversa — não substitui o olhar da equipe."
                                    )

                                m_ficha, m_cen = st.columns((1, 1), gap="small")
                                with m_ficha:
                                    st.metric(
                                        label="Na ficha (original)",
                                        value=f"{proba * 100:.1f}%",
                                        help="Estimativa com os dados tal como estão na base, antes de mudar os controles.",
                                    )
                                with m_cen:
                                    st.metric(
                                        label="Neste cenário (controles)",
                                        value=f"{_sim_pct:.1f}%",
                                        help="Estimativa depois dos valores que você colocou nos deslizadores à esquerda.",
                                    )

                                if abs(delta_pp) < 0.08:
                                    st.markdown(
                                        f"**Em poucas palavras:** **{_sim_pct:.1f}%** com os controles atuais. {_sum_line}"
                                    )
                                else:
                                    st.markdown(
                                        f"**Em poucas palavras:** na ficha (**{proba * 100:.1f}%**) → com os controles (**{_sim_pct:.1f}%**). "
                                        f"{_sum_line}"
                                    )

                                if abs(delta_pp) < 0.08:
                                    if sim_matches_base_ficha(sim, _sim_baseline):
                                        _delta_html = (
                                            "<strong>Igual à ficha:</strong> os controles reproduzem o registo na base — "
                                            "a percentagem coincide. <strong>Altere os indicadores</strong> para explorar outros cenários."
                                        )
                                    else:
                                        _delta_html = (
                                            "<strong>Quase igual à ficha:</strong> a diferença é menor que **0,1** na escala de 0 a 100 — "
                                            "as mudanças nos controles pouco mexeram no resultado para este aluno."
                                        )
                                elif delta_pp > 0:
                                    _delta_html = (
                                        f"<strong>Diferença frente à ficha:</strong> cerca de <strong>{delta_pp:.1f}</strong> pontos "
                                        "a <strong>mais</strong> na escala de 0 a 100 — a leitura automática associa isso a "
                                        "<strong>mais</strong> chance estimada de defasagem ou evasão <em>se</em> estes números "
                                        "refletirem o aluno."
                                    )
                                else:
                                    _delta_html = (
                                        f"<strong>Diferença frente à ficha:</strong> cerca de <strong>{abs(delta_pp):.1f}</strong> pontos "
                                        "a <strong>menos</strong> na escala de 0 a 100 — a leitura automática associa isso a "
                                        "<strong>menos</strong> chance estimada de defasagem ou evasão <em>se</em> estes números "
                                        "refletirem o aluno."
                                    )
                                st.markdown(
                                    f'<div class="pm-risk-delta-box">{_delta_html}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(
                                    "**Dica:** cada «ponto» na diferença = 1 unidade na escala de 0 a 100% (em relatórios técnicos chama-se "
                                    "**ponto percentual**)."
                                )

                                st.markdown(
                                    _risk_subheading_muted(
                                        "Leituras que costumam ajudar",
                                        "Ideias curtas com base nos números do cenário — **sempre** junto da equipe e do que a escola observa.",
                                    ),
                                    unsafe_allow_html=True,
                                )
                                for _line in risk_explain_lines(
                                    float(new_p),
                                    float(sim["Pedra_ord"]),
                                    float(sim["IEG"]),
                                    float(sim["IAA"]),
                                    float(sim["IDA"]),
                                ):
                                    st.markdown(f"- {_line}")

                                _media_tb = _media_turma_base_optional(_eng_row)
                                st.markdown(
                                    _risk_subheading_muted(
                                        "INDE no cenário vs. média dos colegas (base)",
                                        "A média compara com alunos no **mesmo contexto** na base (instituição, fase e ano).",
                                    ),
                                    unsafe_allow_html=True,
                                )
                                if _media_tb is not None:
                                    _ymax = max(10.5, float(sim["INDE"]), float(_media_tb)) * 1.08
                                    fig_cmp = go.Figure(
                                        data=[
                                            go.Bar(
                                                x=["Aluno (simulado)", "Média turma (base)"],
                                                y=[float(sim["INDE"]), float(_media_tb)],
                                                marker_color=["#58a6ff", "#6e7681"],
                                                text=[
                                                    f"{float(sim['INDE']):.2f}",
                                                    f"{float(_media_tb):.2f}",
                                                ],
                                                textposition="outside",
                                            )
                                        ]
                                    )
                                    fig_cmp.update_layout(
                                        template="plotly_dark",
                                        height=300,
                                        margin=dict(l=48, r=28, t=48, b=48),
                                        yaxis=dict(title="INDE", range=[0, _ymax]),
                                        showlegend=False,
                                        paper_bgcolor="rgba(13,17,23,0.25)",
                                        plot_bgcolor="rgba(13,17,23,0.35)",
                                    )
                                    st.plotly_chart(fig_cmp, width="stretch", key=f"pm_rsk_bar_cmp_{ra}")
                                    _gap_ind = float(sim["INDE"]) - float(_media_tb)
                                    _gap_txt = (
                                        f"INDE simulado **{float(sim['INDE']):.2f}** · média do grupo **{float(_media_tb):.2f}** · "
                                        f"**Δ {_gap_ind:+.2f}** em relação à média."
                                    )
                                    if _gap_ind < -0.5:
                                        st.caption(
                                            _gap_txt
                                            + " O aluno simulado fica **claramente abaixo** da média INDE do grupo — possível "
                                            "«peixe fora d'água» no mesmo contexto (instituição/fase/ano na base)."
                                        )
                                    elif _gap_ind > 0.5:
                                        _above = _gap_txt + " **Acima da média do grupo** neste recorte."
                                        if float(new_p) > OPERATIONAL_HIGH_RISK_THRESHOLD:
                                            _above += (
                                                " O modelo **não resume só ao INDE** — outros fatores entram na probabilidade; "
                                                "cruze com o SHAP e o olhar da equipe."
                                            )
                                        st.caption(_above)
                                    else:
                                        st.caption(_gap_txt + " **Próximo da média do grupo.**")
                                else:
                                    st.caption(
                                        "Média da turma indisponível neste registro — não foi possível exibir o gráfico de barras da comparação."
                                    )

        else:
            st.info(
                "**Próximo passo:** escolhe um aluno na caixa acima (podes escrever ou colar o RA) para ver risco na ficha, "
                "SHAP e parecer do Theo (e, se precisar, simulação no modo técnico)."
            )

    with tab2:
        st.markdown(
            _risk_heading_html(
                "Matriz de priorização",
                "Lista do **maior** para o **menor** risco (probabilidade do modelo). Filtra por **ano** (por defeito 2024), **fase** (1–8) e **turma** (A–E); "
                "só entram os dados que estão carregados agora no app.",
                tight_top=True,
            ),
            unsafe_allow_html=True,
        )
        with st.expander("O que essa tela faz (e o que ela não substitui)", expanded=False):
            st.markdown(
                f"- Aqui a gente usa **{OPERATIONAL_HIGH_RISK_THRESHOLD:.0%}** como marca de **alto risco** na interface — foi assim que o modelo foi calibrado pra operar no dia a dia.\n"
                "- Isso **ajuda** na triagem, mas **não troca** regra da escola, ata de CIEP, lista oficial nem o que a equipe já sabe do aluno na prática.\n"
                "- **Ano de referência na lista:** por defeito **2024** (se existir na base); **Todos os anos** volta ao recorte sem filtro de ano. Com um ano fixo + **um registo por aluno**, a percentagem **bate com a ficha** da outra aba se escolheres o **mesmo ano** lá.\n"
                "- Com **«Todos os registos do recorte»** (checkbox desligado) e ano **Todos**, vês todas as linhas; o mesmo RA pode ter riscos diferentes por ano.\n"
                "- Com **«Um registo por aluno»** ligado e ano **Todos**, usa-se o **último ano global** por RA e depois fase/turma.\n"
                "- A ordem vem **só do modelo** em cima do Parquet que tá aberto — não é documento de encaminhamento nem decisão automática."
            )

        if not ranking_fase_options(df):
            st.warning("Não há valores de **Fase** reconhecíveis (1–8) na base carregada.")
        else:
            st.markdown(
                _risk_subheading_muted(
                    "Critérios de recorte",
                    "Ano de referência (por defeito 2024), depois fase (ou **todas**) e turma — só aparecem turmas que **existem** no recorte atual.",
                ),
                unsafe_allow_html=True,
            )
            _has_ref_year = ("Ano" in df.columns) or ("ano_referencia" in df.columns)
            _years_tab2 = reference_years_available(df) if _has_ref_year else []
            sel_ref_year: int | None
            if _years_tab2:
                _year_opts: list[int | None] = _years_tab2 + [None]
                _def_y_idx = _year_opts.index(2024) if 2024 in _year_opts else 0
                sel_ref_year = st.selectbox(
                    "Ano de referência na lista",
                    options=_year_opts,
                    index=_def_y_idx,
                    format_func=lambda x: "Todos os anos" if x is None else str(int(x)),
                    key="pm_rsk_tab2_ref_year",
                    help="**2024** (se existir) como vista operacional; **Todos** não filtra por ano. Para bater com a ficha: o **mesmo** ano na outra aba + um registo por aluno.",
                )
            else:
                sel_ref_year = None
                st.caption("Sem coluna **Ano** / **ano_referencia** — não dá para filtrar a lista por ano.")
            _one_per_ra = st.checkbox(
                "Um registo por aluno — **igual à ficha individual** (no ano escolhido)",
                value=True,
                key="pm_rsk_tab2_one_row_per_ra",
                help="Com **ano fixo** (ex. 2024): um aluno = uma linha desse ano (bate com a ficha desse ano). Com **Todos os anos** + ligado: último ano **global** por RA; depois fase/turma. "
                "Desligado: todas as linhas do recorte de ano — pode duplicar RA.",
            )
            if sel_ref_year is None:
                _df_basis = latest_row_per_ra_table(df.copy()) if _one_per_ra else df.copy()
            else:
                _yi = int(sel_ref_year)
                _df_basis = (
                    one_row_per_ra_for_year(df.copy(), _yi)
                    if _one_per_ra
                    else rows_for_reference_year(df.copy(), _yi)
                )
            fases_opts = ranking_fase_options(_df_basis) or ranking_fase_options(df)
            fase_ui_options: list[int | None] = [None] + fases_opts
            if _one_per_ra and sel_ref_year is not None:
                st.caption(
                    f"**Fase** e **turma** são as do registo de **{int(sel_ref_year)}** — alinha com a ficha se escolheres o mesmo ano na **Análise individual**."
                )
            elif _one_per_ra:
                st.caption(
                    "**Fase** e **turma** do **último ano global** por aluno — na outra aba escolhe o **ano mais recente** da lista para comparar."
                )
            c1, c2, c3 = st.columns([1.15, 1.15, 1.0])
            with c1:
                f_sel = st.selectbox(
                    "Fase escolar",
                    fase_ui_options,
                    index=0,
                    format_func=lambda x: "Todas as fases" if x is None else f"Fase {int(x)}",
                    key="pm_rsk_tab2_fase",
                    help="Com **Todas**, não filtra por fase; com uma fase fixa, a lista de turmas só mostra letras que aparecem nela.",
                )
            turmas_opts = ranking_turma_letter_options_for_fase(_df_basis, f_sel)
            turmas_ui = [""] + turmas_opts
            _turma_key = f"pm_rsk_tab2_turma__{f_sel if f_sel is not None else 'all'}"
            with c2:
                t_sel = st.selectbox(
                    "Turma",
                    turmas_ui,
                    index=0,
                    format_func=lambda x: "Todas" if x == "" else f"Turma {x}",
                    key=_turma_key,
                    help="Só letras que existem na base **nessa fase** (ou em todas, se fase = Todas).",
                )
            with c3:
                profundidade_opts: list[int | None] = [10, 25, 50, 100, None]
                topn = st.selectbox(
                    "Profundidade da lista",
                    profundidade_opts,
                    index=0,
                    key="pm_rsk_tab2_top",
                    format_func=lambda x: "Todas (recorte inteiro)" if x is None else f"{int(x)} linhas",
                    help="Limite de linhas na tabela ou **recorte inteiro** — sempre ordenado do maior para o menor risco no filtro atual.",
                )

            if f_sel is not None and t_sel and turmas_opts and t_sel not in turmas_opts:
                st.caption(
                    f"A turma **{t_sel}** não aparece na fase **{int(f_sel)}** nesse arquivo — escolhe outra letra ou **Todas**."
                )

            mask = ranking_mask(_df_basis, f_sel, str(t_sel))
            _slice_df = _df_basis.loc[mask].copy()
            full_ranked = predict_risk_slice(bundle, _slice_df)
            n_recorte = len(full_ranked)
            if full_ranked.empty:
                if sel_ref_year is not None:
                    st.warning(
                        f"**Nenhum registo** para o ano **{int(sel_ref_year)}** com este recorte (fase/turma). "
                        "Tenta **Todos os anos**, outra fase ou turma **Todas**."
                    )
                else:
                    st.warning(
                        "**Não achou ninguém** com esse filtro. Tenta deixar **Todas** nas turmas, mudar a fase "
                        "ou conferir se o Parquet tem dado mesmo pra esse recorte."
                    )
            else:
                mean_p = float(full_ranked["risco"].mean())
                pct_hi = float((full_ranked["risco"] >= OPERATIONAL_HIGH_RISK_THRESHOLD).mean() * 100.0)
                mx = float(full_ranked["risco"].max())
                ranked = full_ranked if topn is None else full_ranked.head(int(topn))
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Registros no recorte", f"{n_recorte:,}".replace(",", "."))
                with m2:
                    st.metric("Probabilidade média", f"{mean_p * 100:.1f}%")
                with m3:
                    st.metric(
                        f"≥ limiar ({OPERATIONAL_HIGH_RISK_THRESHOLD:.0%})",
                        f"{pct_hi:.1f}%",
                        help="% de linhas no recorte com probabilidade igual ou acima do limiar de alto risco.",
                    )
                with m4:
                    st.metric("Máximo no recorte", f"{mx * 100:.1f}%")

                st.divider()
                n_show = len(ranked)
                _n_pt = lambda n: f"{int(n):,}".replace(",", ".")
                if topn is None:
                    prior_txt = (
                        f"Mostrando **todos os {_n_pt(n_show)}** registros do recorte "
                        "— ordem do modelo do maior para o menor risco."
                    )
                else:
                    prior_txt = (
                        f"Mostrando **{_n_pt(n_show)}** linhas (limite **{int(topn)}**) de **{_n_pt(n_recorte)}** no recorte "
                        "— quem tá mais acima na fila do modelo."
                    )
                st.markdown(
                    _risk_subheading_muted(
                        "Priorização",
                        prior_txt,
                    ),
                    unsafe_allow_html=True,
                )

                show = ranked.copy()
                show["Probabilidade (%)"] = (show["risco"] * 100).round(1)
                out = show[["RA", "Nome", "Fase", "Turma", "Ano", "Probabilidade (%)"]].copy()
                st.dataframe(
                    out,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "RA": st.column_config.TextColumn("RA", help="Identificador do aluno no arquivo."),
                        "Nome": st.column_config.TextColumn("Nome", width="large"),
                        "Fase": st.column_config.NumberColumn("Fase", format="%d", help="Fase do programa."),
                        "Turma": st.column_config.TextColumn("Turma", help="Turma (A–E) no dado."),
                        "Ano": st.column_config.NumberColumn("Ano ref.", format="%d", help="Ano de referência da linha."),
                        "Probabilidade (%)": st.column_config.ProgressColumn(
                            "Prob. alto risco",
                            help="Chance estimada pelo modelo (0–100%). Quanto mais cheia a barra, maior a prioridade relativa nesse filtro.",
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                )

                csv_buf = io.StringIO()
                out.to_csv(csv_buf, index=False, encoding="utf-8-sig", sep=";")
                st.download_button(
                    label="Exportar lista visível (CSV)",
                    data=csv_buf.getvalue().encode("utf-8-sig"),
                    file_name=(
                        f"priorizacao_risco_ano{int(sel_ref_year) if sel_ref_year is not None else 'todos'}"
                        f"_fase{'todas' if f_sel is None else int(f_sel)}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
                    ),
                    mime="text/csv",
                    key="pm_rsk_tab2_export_csv",
                    help="Mesmas colunas da tabela; separador `;`; UTF-8 com BOM (abre certinho no Excel).",
                )
                st.caption(
                    "Dica: cruza essa ordem com a **ficha** na outra aba e com o que a **equipe** já sabe do aluno antes de bater o martelo."
                )


def render_dictionary(df: pd.DataFrame) -> None:
    st.subheader("Dicionário de dados")
    st.caption(
        "Colunas e ordem vêm do **Parquet carregado** (mesmo ficheiro do chat e dos dashboards). "
        "Edite as descrições e clique em **Salvar dicionário**; o Theo usa este texto no SQL e nas respostas."
    )
    rows = merge_dictionary_with_dataframe(df, load_dictionary())
    edited = st.data_editor(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "coluna": st.column_config.TextColumn("Coluna"),
            "descricao": st.column_config.TextColumn("Descrição", width="large"),
        },
    )
    if st.button("Salvar dicionário"):
        save_dictionary(edited.to_dict("records"))
        st.toast("Novo contexto aplicado ao Theo no próximo envio do chat.", icon="✅")

    csv_path = DATA_DIR / "relatorio.csv"
    pq_path = get_parquet_path()
    try:
        pq_rel = pq_path.resolve().relative_to(_ROOT.resolve())
        pq_show = str(pq_rel).replace("\\", "/")
    except ValueError:
        pq_show = str(pq_path.resolve())
    try:
        csv_rel = csv_path.resolve().relative_to(_ROOT.resolve())
        csv_show = str(csv_rel).replace("\\", "/")
    except ValueError:
        csv_show = str(csv_path.resolve())

    with st.expander("Pré-visualizar dados (Parquet em uso)", expanded=False):
        st.markdown(
            "Isto é o **mesmo** `DataFrame` do chat, risco e dashboards (após normalização). "
            "O dicionário acima alinha-se a estas colunas — **não** ao CSV bruto."
        )
        st.caption(f"Parquet: `{pq_show}`")
        n_preview = st.slider("Linhas a mostrar", 25, 500, 100, 25, key="dict_preview_n")
        view = df.head(n_preview)
        st.dataframe(view, width="stretch", hide_index=True)
        st.caption(
            f"Total na base carregada: **{len(df):,}** linhas × **{len(df.columns)}** colunas (a mostrar {min(n_preview, len(df)):,}).".replace(
                ",", "."
            )
        )
        sample_csv = view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descarregar amostra (CSV)",
            data=sample_csv,
            file_name="amostra_parquet_carregado.csv",
            mime="text/csv",
            key="dict_dl_parquet_sample",
        )
        with st.expander("Opcional: pré-visualizar `relatorio.csv` (entrada ao ETL)", expanded=False):
            st.caption(
                "Só para confrontar com o export original; colunas podem estar em minúsculas e **não** espelham o DuckDB `dados`."
            )
            if not csv_path.exists():
                st.info(f"Não há ficheiro em `{csv_show}`.")
            else:
                st.caption(f"Ficheiro: `{csv_show}`")
                try:
                    csv_df = pd.read_csv(csv_path, nrows=n_preview, encoding="utf-8-sig", low_memory=False)
                except UnicodeDecodeError:
                    csv_df = pd.read_csv(csv_path, nrows=n_preview, encoding="latin-1", low_memory=False)
                except Exception as exc:
                    st.error(f"Não foi possível ler o CSV: {exc}")
                else:
                    st.dataframe(csv_df, width="stretch", hide_index=True)
                    st.caption(
                        f"Mostrando **{len(csv_df):,}** linhas lidas do CSV".replace(",", ".")
                        + (" (limite do controle acima)." if len(csv_df) < n_preview else ".")
                    )
                    st.download_button(
                        "Descarregar esta pré-visualização (CSV)",
                        data=csv_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="amostra_relatorio_csv.csv",
                        mime="text/csv",
                        key="dict_dl_csv_sample",
                    )


def _embed_annual_site(url: str) -> None:
    """Mostra o relatório no Streamlit com o mesmo estilo de página cheia que o PDF tinha."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        st.error("URL do relatório inválido (é preciso http ou https).")
        return

    _frame_h = min(960, max(520, int(900 * 0.88)))
    safe_url = html.escape(url.strip(), quote=True)
    iframe = (
        f'<iframe src="{safe_url}" width="100%" height="100%" '
        f'style="min-height:{_frame_h}px;border:none;border-radius:6px;" '
        f'title="Relatório anual" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>'
    )
    components.html(iframe, height=_frame_h + 32, scrolling=True)


def render_annual_report() -> None:
    st.title("Relatório anual")
    _embed_annual_site(_annual_gamma_url())


def main() -> None:
    _icon = _ROOT / "assets" / "app_icon.png"
    page_icon: str | Path = str(_icon) if _icon.exists() else "🎓"
    st.set_page_config(
        page_title="Passos Mágicos — Painel local",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sidebar_page" not in st.session_state:
        st.session_state.sidebar_page = "chat"

    ollama_ok = ollama_available()
    df = cached_load_dados()
    bundle = cached_model_bundle()
    dict_rows = merge_dictionary_with_dataframe(df, load_dictionary())
    dictionary_block = rows_to_prompt_block(dict_rows)
    annual_plain = _annual_plain_for_theo()
    theo_context_block = merge_theo_context_blocks(
        dictionary_block, annual_plain, load_gamma_context_text()
    )

    with st.sidebar:
        # JPEG não suporta transparência: o “xadrez” do editor vira branco/cinza sólido no ficheiro.
        # Para fundo transparente na sidebar, exporte PNG com canal alpha como assets/slogan.png (tem prioridade).
        _slogan_png = _ROOT / "assets" / "slogan.png"
        _slogan_jpeg = _ROOT / "assets" / "slogan.jpeg"
        _slogan_path = (
            _slogan_png
            if _slogan_png.exists()
            else (_slogan_jpeg if _slogan_jpeg.exists() else None)
        )
        _brand_copy_html = (
            '<div class="pm-sidebar-brand-text pm-brand-hero pm-sidebar-brand-copy">'
            '<p class="pm-sidebar-tagline">Transformando dados em decisões inteligentes</p>'
            '<p class="pm-branded-subkicker">PAINEL LOCAL</p>'
            '<p class="pm-sidebar-hint">Painel fechado? Use o botão <strong>☰</strong> na '
            "<strong>barra superior</strong> (ao lado do título da página) para abrir o menu.</p>"
            "</div>"
        )
        if _slogan_path is not None:
            _uri = _sidebar_slogan_data_uri(_slogan_path)
            if _uri:
                st.markdown(
                    f'<div class="pm-sidebar-brand-layout">'
                    f'<div class="pm-sidebar-brand-stage">'
                    f'<div class="pm-sidebar-brand-visual">'
                    f'<img src="{_uri}" alt="" class="pm-sidebar-slogan-hero" loading="lazy" />'
                    f'<div class="pm-sidebar-hands-glow" aria-hidden="true"></div>'
                    f"</div>"
                    f"{_brand_copy_html}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                _sc1, _sc2, _sc3 = st.columns([0.12, 0.76, 0.12])
                with _sc2:
                    st.image(str(_slogan_path), use_container_width=True)
                    st.markdown(_brand_copy_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="pm-sidebar-brand-layout"><div class="pm-sidebar-brand-stage">'
                f"{_brand_copy_html}</div></div>",
                unsafe_allow_html=True,
            )
        ollama_class = "pm-ollama-ok" if ollama_ok else "pm-ollama-off"
        ollama_txt = "Ollama conectado — LLM local" if ollama_ok else "Ollama indisponível — inicie o serviço"
        st.markdown(
            f'<div class="pm-ollama-pill">'
            f'<span class="pm-ollama-dot {ollama_class}"></span>'
            f"<span>{ollama_txt}</span></div>",
            unsafe_allow_html=True,
        )
        page_ids = [p[0] for p in PAGE_DEFS]
        current = st.session_state.sidebar_page
        if current not in page_ids:
            current = "chat"
            st.session_state.sidebar_page = current
        page = st.radio(
            "\u200b",
            options=page_ids,
            index=page_ids.index(current),
            format_func=lambda pid: PAGE_LABELS[pid],
            label_visibility="hidden",
            key="sidebar_nav_radio",
        )
        st.session_state.sidebar_page = page
        st.markdown(f'<div class="pm-tool-help">{PAGE_HELP[page]}</div>', unsafe_allow_html=True)
        if st.button("Limpar histórico do chat", width="stretch"):
            st.session_state.messages = []
            st.toast("Histórico limpo.", icon="🧹")
        st.caption("Made by Joe Allan Zirn")

    df_sql, sql_runner = make_chat_sql_runner(df, bundle)

    if page == "chat":
        render_chat(theo_context_block, ollama_ok, sql_runner, sql_context_df=df_sql)
    elif page == "annual_report":
        render_annual_report()
    elif page == "risk":
        render_risk(df, bundle, theo_context_block)
    elif page == "dashboard":
        render_dashboards(df, bundle)
    else:
        render_dictionary(df)


if __name__ == "__main__":
    main()
