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
import io
import json
import os
import sys
from pathlib import Path

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
from passos_magico.llm.ml_text import generate_diagnosis_text
from passos_magico.llm.ollama_client import ollama_available
from passos_magico.llm.institutional_router import is_institutional_narrative_only
from passos_magico.llm.pipeline import (
    invoke_institutional_insight,
    sql_and_chart_step,
    stream_institutional_insight_text,
    stream_insight_text,
    suggestions_step,
)
from passos_magico.ml.features import (
    FEATURE_ORDER,
    latest_row_per_ra_table,
    pick_latest_year_row,
    row_features_from_df,
)
from passos_magico.ml.inference import (
    explain_row_shap,
    predict_risk_batch,
    predict_row_after_simulation,
    predict_row_features,
)
from passos_magico.ml.risk_display import OPERATIONAL_HIGH_RISK_THRESHOLD
from passos_magico.llm.charts import CHART_TYPE_OPTIONS, figure_from_dataframe, heuristic_kind_to_chart_id
from passos_magico.semantic.metadata import (
    load_annual_summary_text,
    load_dictionary,
    load_gamma_context_text,
    merge_theo_context_blocks,
    rows_to_prompt_block,
    save_dictionary,
)
from passos_magico.ui.dashboard import render_dashboards
from passos_magico.ui.risk_plan_suggest import suggest_minimal_ieg_ida
from passos_magico.ui.risk_sim_copy import risk_explain_lines, sim_matches_base_ficha, snapshot_sim_baseline
from passos_magico.ui.ranking_filters import (
    ranking_fase_options,
    ranking_mask,
    ranking_turma_letter_options,
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
        "O PDF abre nesta página com o leitor do Streamlit. Ficheiro: assets/relatorio_anual.pdf. "
        "Requer o pacote **streamlit-pdf** (`pip install streamlit-pdf`). Opcional: URL Gamma.",
    ),
    (
        "risk",
        "🎯  Previsão de risco",
        "Um aluno de cada vez: risco na ficha, contexto pedagógico, SHAP e parecer do Theo; simulação manual só no modo técnico (opcional). "
        "Segunda aba: lista ordenada por risco com filtros de fase (1–8) e turma (A–E), sem duplicados por grafia.",
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
        "Salve para aplicar na próxima pergunta do chat. Na mesma página pode pré-visualizar o Parquet em uso ou o CSV em data/relatorio.csv.",
    ),
]
PAGE_LABELS: dict[str, str] = {p[0]: p[1] for p in PAGE_DEFS}
PAGE_HELP: dict[str, str] = {p[0]: p[2] for p in PAGE_DEFS}
_MAX_CHART_ROWS = 5000

_ANNUAL_PDF_PATH = _ROOT / "assets" / "relatorio_anual.pdf"


def _annual_gamma_url() -> str:
    """URL público da apresentação Gamma (opcional). Ambiente tem prioridade sobre secrets."""
    env = os.environ.get("PM_RELATORIO_ANUAL_GAMMA_URL", "").strip()
    if env:
        return env
    try:
        v = st.secrets["RELATORIO_ANUAL_GAMMA_URL"]
    except Exception:
        return ""
    return str(v).strip() if v else ""


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
        "O Theo gera SQL, mostra um gráfico e explica. Pode mudar o tipo de gráfico abaixo da figura sem repetir a pergunta."
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
            err_msg = f"**Theo:** Ocorreu um erro inesperado: `{e!s}`"
            st.markdown(err_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "id": msg_id,
                    "content": err_msg,
                    "sql": None,
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
        sugs = suggestions_step(user_text, insight_full)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "id": msg_id,
                "content": insight_full,
                "sql": "-- Pergunta respondida com o texto do resumo anual (sem consulta à tabela `dados`).",
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
        elif ollama_ok:
            text = f"**Theo:** Não consegui concluir a análise. Detalhes: {err}"
        else:
            text = "**Theo:** Ollama indisponível. Inicie o serviço local e tente novamente."
        st.markdown(text)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "id": msg_id,
                "content": text,
                "sql": step.sql,
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
    sugs = suggestions_step(user_text, insight_full)
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


def _risk_plan_session_key(ra: str, sim: dict[str, Any]) -> str:
    """Chave estável para guardar sugestão de plano enquanto o cenário não muda."""
    payload = {k: round(float(sim[k]), 3) for k in sorted(sim.keys())}
    return f"{ra}|{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"


def _get_engineered_row_for_display(df: pd.DataFrame, ra: str) -> pd.Series | None:
    """Mesma linha que `row_features_from_df`, com colunas de engenharia do modelo de risco."""
    from passos_magico.ml.risk_pipeline import ensure_risk_engineering

    ra_col = "RA" if "RA" in df.columns else "ra"
    if ra_col not in df.columns:
        return None
    sub = df[df[ra_col].astype(str) == str(ra)]
    if sub.empty:
        return None
    key = pick_latest_year_row(sub).iloc[0]
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
        "Não substitui a equipe. **Aba 1:** um aluno (ficha, contexto, SHAP, Theo). **Aba 2:** ranking. "
        "**Simulação** só no modo técnico."
    )
    with st.expander("Ajuda rápida (SHAP, 46%, simulação)", expanded=False):
        st.markdown(
            "- **Risco:** estimativa do modelo sobre a ficha; **≤46%** baixo, **46–65%** atenção, **>65%** mais acompanhamento.\n"
            "- **SHAP:** barra à **direita** da linha = aumenta o risco neste modelo; à **esquerda** = reduz.\n"
            "- **Theo:** usa risco da ficha + SHAP; não inventa percentagens.\n"
            "- **Modo técnico:** cenário «e se…»; a base **não** é alterada."
        )

    tab1, tab2 = st.tabs(["Análise individual", "Ranking (triagem)"])

    with tab1:
        st.markdown(
            _risk_heading_html(
                "1 · Quem analisar",
                "Indique o RA ou escolha o nome na lista.",
                tight_top=True,
            ),
            unsafe_allow_html=True,
        )
        _picker_df = latest_row_per_ra_table(df)
        _nome_col = "Nome" if "Nome" in _picker_df.columns else "nome"
        _ra_col_p = "RA" if "RA" in _picker_df.columns else "ra"
        names = _picker_df[_nome_col].astype(str) + " — " + _picker_df[_ra_col_p].astype(str)
        ra_from_select = None
        c1, c2 = st.columns(2)
        with c1:
            ra_input = st.text_input(
                "Registro do aluno (RA)",
                placeholder="RA2020001",
                help="Igual ao que aparece na base. Pode copiar e colar.",
            )
        with c2:
            pick = st.selectbox(
                "Ou escolha pelo nome na lista",
                options=[""] + list(names),
                help="Lista com um registro por aluno (último ano de referência na base).",
            )
        if pick:
            ra_from_select = str(_picker_df.loc[names == pick, _ra_col_p].iloc[0])
        ra = (ra_from_select or ra_input or "").strip()
        feats = row_features_from_df(df, ra) if ra else None

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

            _eng_row = _get_engineered_row_for_display(df, ra)
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
                diag = generate_diagnosis_text(
                    nome,
                    ra,
                    proba,
                    shap_pairs,
                    theo_context_block=theo_context_block,
                    feats=feats,
                )
                st.markdown(diag)

            with st.expander("Modo técnico: simulação manual (opcional)", expanded=False):
                st.markdown(
                    "**Aviso:** a simulação altera só os indicadores dos controlos; outras variáveis do modelo podem "
                    "permanecer alinhadas à **última ficha** na base — o resultado é **exploratório**, não substitui "
                    "o risco oficial da secção 2. A base de dados **não** é alterada."
                )
                _sim_active = st.checkbox(
                    "Ativar simulador (mostra controlos e recalcula a probabilidade)",
                    value=False,
                    key=f"pm_rsk_sim_on_{ra}",
                )
                if not _sim_active:
                    st.caption(
                        "Ligue a opção acima para abrir o painel de controlos e comparar cenários com a ficha."
                    )
                else:
                    st.markdown(
                        _risk_heading_html(
                            "Simulador «e se…?»",
                            "Esquerda: controlos (0–10). Direita: novo risco e comparação. IPV vem da base.",
                            tight_top=True,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Cada alteração recalcula o modelo (IAA−IDA, distância à média da turma, Δ INDE como no treino). "
                        "A base **não** é alterada."
                    )

                    sim = dict(base_feats)
                    sim["IAA"] = _eng_float(_eng_row, "iaa", 5.0)
                    sim["IPS"] = _eng_float(_eng_row, "ips", 5.0)
                    sim["MAT"] = _eng_float(_eng_row, "mat", 5.0)
                    sim["POR"] = _eng_float(_eng_row, "por", 5.0)
                    sim["Delta_INDE"] = _eng_float(_eng_row, "delta_inde", 0.0)
        
                    _sim_baseline = snapshot_sim_baseline(sim)
        
                    col_ctrl, col_main = st.columns([0.38, 0.62], gap="medium")
        
                    with col_ctrl:
                        st.markdown("**Atitude (impacto alto)**")
                        sim["IEG"] = st.slider(
                            "IEG — Engajamento", 0.0, 10.0, float(sim["IEG"]), 0.1, key="pm_rsk_ieg"
                        )
                        sim["IAA"] = st.slider(
                            "IAA — Autoavaliação", 0.0, 10.0, float(sim["IAA"]), 0.1, key="pm_rsk_iaa"
                        )
                        sim["IPS"] = st.slider(
                            "IPS — Psicossocial", 0.0, 10.0, float(sim["IPS"]), 0.1, key="pm_rsk_ips"
                        )
        
                        st.markdown("**Acadêmico**")
                        sim["INDE"] = st.slider("INDE", 0.0, 10.0, float(sim["INDE"]), 0.1, key="pm_rsk_inde")
                        sim["IDA"] = st.slider(
                            "IDA — Aprendizagem (nota real)", 0.0, 10.0, float(sim["IDA"]), 0.1, key="pm_rsk_ida"
                        )
                        sim["IAN"] = st.slider("IAN", 0.0, 10.0, float(sim["IAN"]), 0.1, key="pm_rsk_ian")
                        sim["MAT"] = st.slider("MAT", 0.0, 10.0, float(sim["MAT"]), 0.1, key="pm_rsk_mat")
                        sim["POR"] = st.slider("POR", 0.0, 10.0, float(sim["POR"]), 0.1, key="pm_rsk_por")
        
                        st.markdown("**Trajetória**")
                        sim["Delta_INDE"] = st.slider(
                            "Evolução esperada (Δ INDE)",
                            -5.0,
                            5.0,
                            float(sim["Delta_INDE"]),
                            0.1,
                            key="pm_rsk_delta_inde",
                            help="Tendência de melhora ou queda do INDE no recurso «delta_inde» do modelo — frequentemente pesa junto com o INDE instantâneo.",
                        )
                        st.caption("Δ INDE mede evolução esperada; pode compensar um INDE instantâneo mais baixo.")
                        _choque_ui = float(sim["IAA"]) - float(sim["IDA"])
                        st.caption(f"**Choque de realidade** (IAA − IDA) neste cenário: **{_choque_ui:+.1f}**.")
        
                        st.markdown("**Contexto (ficha)**")
                        st.caption(
                            f"Fase {int(base_feats['Fase'])} · Pedra {_pedra_ord_to_name(base_feats['Pedra_ord'])} · "
                            f"Turma {_turma_ord_to_letter(base_feats['Turma_ord'])} · Ano {int(base_feats['Ano'])}"
                        )
                        with st.expander("Notas sobre Fase 8 e Pedra Quartzo no modelo", expanded=False):
                            st.markdown(
                                "No histórico usado no treino, combinações com **Fase 8** ou **Pedra Quartzo** tendem a estar "
                                "associadas a maior probabilidade de risco no modelo — não são juízos sobre o aluno, "
                                "mas contexto estatístico."
                            )
                        if int(round(base_feats["Fase"])) == 8:
                            st.warning(
                                "**Fase 8** neste registro — contexto que o modelo trata como sensível."
                            )
                        if int(round(base_feats["Pedra_ord"])) == 1:
                            st.warning(
                                "**Pedra Quartzo** neste registro — nível associado a maior risco no histórico treinado."
                            )
        
                        with st.expander(
                            "Cenário avançado: mudar fase, turma, ano ou pedra (opcional)",
                            expanded=False,
                        ):
                            st.caption(
                                "Só para simulações. Para acompanhar o registro atual, deixe fechado ou volte aos valores da base."
                            )
                            sim["Fase"] = st.slider(
                                "Fase do programa",
                                1.0,
                                8.0,
                                float(sim["Fase"]),
                                1.0,
                                key="pm_rsk_fase",
                                help="Número da fase (1 a 8), como no relatório.",
                            )
                            sim["Turma_ord"] = st.slider(
                                "Turma (1 = A … 5 = E)",
                                1.0,
                                5.0,
                                float(sim["Turma_ord"]),
                                1.0,
                                key="pm_rsk_turma",
                                help="Use o número correspondente à letra da turma na escola.",
                            )
                            sim["Ano"] = st.slider(
                                "Ano de referência",
                                float(df["Ano"].min()),
                                float(df["Ano"].max()),
                                float(sim["Ano"]),
                                1.0,
                                key="pm_rsk_ano",
                            )
                            sim["Pedra_ord"] = st.slider(
                                "Pedra (nível de desempenho)",
                                1.0,
                                4.0,
                                float(sim["Pedra_ord"]),
                                1.0,
                                key="pm_rsk_pedra",
                                help="1 = Quartzo, 2 = Ágata, 3 = Ametista, 4 = Topázio.",
                            )
        
                        _ctx_changed = (
                            int(round(sim["Fase"])) != int(round(base_feats["Fase"]))
                            or int(round(sim["Turma_ord"])) != int(round(base_feats["Turma_ord"]))
                            or int(round(sim["Ano"])) != int(round(base_feats["Ano"]))
                            or int(round(sim["Pedra_ord"])) != int(round(base_feats["Pedra_ord"]))
                        )
                        _ctx_note = (
                            " _(fase / turma / ano / pedra alterados no cenário avançado)_"
                            if _ctx_changed
                            else " _(contexto = base)_"
                        )
                        st.caption(
                            f"**Resumo do cenário**{_ctx_note}: "
                            f"Fase **{int(sim['Fase'])}** · turma **{_turma_ord_to_letter(sim['Turma_ord'])}** "
                            f"· ano **{int(sim['Ano'])}** · pedra **{_pedra_ord_to_name(sim['Pedra_ord'])}** · "
                            f"INDE **{sim['INDE']:.1f}** · IDA **{sim['IDA']:.1f}** · IAN **{sim['IAN']:.1f}** · "
                            f"IEG **{sim['IEG']:.1f}** · IAA **{sim['IAA']:.1f}** · IPS **{sim['IPS']:.1f}** · "
                            f"MAT **{sim['MAT']:.1f}** · POR **{sim['POR']:.1f}** · ΔINDE **{sim['Delta_INDE']:+.1f}** · "
                            f"IPV (base) **{sim['IPV']:.1f}**"
                        )
        
                    new_p = predict_row_after_simulation(bundle, df, ra, sim)
        
                    with col_main:
                        if new_p is None or (isinstance(new_p, float) and np.isnan(new_p)):
                            st.error("Não foi possível calcular o risco simulado para este RA.")
                        else:
                            delta_pp = (new_p - proba) * 100.0
                            with st.container(border=True):
                                st.markdown("**Resultado da simulação**")
                                crit46 = float(new_p) > OPERATIONAL_HIGH_RISK_THRESHOLD
                                th_txt = (
                                    "Acima do limiar de 46% (atenção)."
                                    if crit46
                                    else "Igual ou abaixo de 46%."
                                )
                                if crit46:
                                    st.warning(f"Limiar 46%: {th_txt}")
                                else:
                                    st.success(f"Limiar 46%: {th_txt}")
                                st.metric(
                                    "Probabilidade simulada de alto risco",
                                    f"{float(new_p) * 100:.1f}%",
                                )
                                _sim_pct = float(new_p) * 100.0
                                if abs(delta_pp) < 0.08:
                                    _sum_line = (
                                        "**praticamente igual** à probabilidade da ficha na base."
                                        if sim_matches_base_ficha(sim, _sim_baseline)
                                        else "**quase igual** à da ficha — os controlos mudaram pouco o resultado."
                                    )
                                elif delta_pp > 0:
                                    _sum_line = (
                                        f"**sobe cerca de {delta_pp:.1f} pontos** na escala de 0 a 100 em relação à ficha "
                                        "(o modelo lê **mais** alerta)."
                                    )
                                else:
                                    _sum_line = (
                                        f"**desce cerca de {abs(delta_pp):.1f} pontos** na escala de 0 a 100 em relação à ficha "
                                        "(o modelo lê **menos** alerta)."
                                    )
                                st.caption(
                                    f"Com os números que pôs nos controlos, a ferramenta estima **{_sim_pct:.1f}%** "
                                    f"de probabilidade de alto risco — {_sum_line}"
                                )
        
                                _cap_f8 = int(round(sim["Fase"])) == 8
                                _cap_qz = int(round(sim["Pedra_ord"])) == 1
                                if _cap_f8 or _cap_qz:
                                    if _cap_f8 and _cap_qz:
                                        _cap_ctx = (
                                            "No cenário atual, **Fase 8** e **Pedra Quartzo** reforçam o contexto de risco no modelo — "
                                            "combine com os indicadores ajustados."
                                        )
                                    elif _cap_f8:
                                        _cap_ctx = (
                                            "No cenário atual, **Fase 8** reforça o contexto de risco no modelo — "
                                            "combine com os indicadores ajustados."
                                        )
                                    else:
                                        _cap_ctx = (
                                            "No cenário atual, **Pedra Quartzo** reforça o contexto de risco no modelo — "
                                            "combine com os indicadores ajustados."
                                        )
                                    st.caption(_cap_ctx)
        
                            with st.container(border=True):
                                st.markdown(
                                    _risk_subheading_muted(
                                        "Interpretação rápida",
                                        "O que estes números sugerem, em linguagem simples (use também o gráfico de fatores acima).",
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
        
                                if "pm_rsk_plan_suggest" not in st.session_state:
                                    st.session_state.pm_rsk_plan_suggest = {"key": "", "result": None}
                                _plan_key = _risk_plan_session_key(ra, sim)
        
                                st.caption(
                                    "A sugestão abaixo só altera **IEG** (engajamento) e **IDA** (aprendizagem) no modelo; "
                                    "os outros controlos ficam fixos neste cálculo."
                                )
                                if st.button("Sugerir plano de ação", key=f"pm_rsk_suggest_plan_{ra}"):
        
                                    def _plan_predict(s: dict[str, Any]) -> float:
                                        return float(predict_row_after_simulation(bundle, df, ra, s))
        
                                    st.session_state.pm_rsk_plan_suggest = {
                                        "key": _plan_key,
                                        "result": suggest_minimal_ieg_ida(
                                            sim, _plan_predict, threshold=OPERATIONAL_HIGH_RISK_THRESHOLD, step=0.25
                                        ),
                                    }
        
                                _plan_cache = st.session_state.pm_rsk_plan_suggest
                                if _plan_cache["key"] == _plan_key and _plan_cache["result"] is not None:
                                    _pres = _plan_cache["result"]
                                    if _pres.status == "already_below":
                                        st.info(
                                            "Neste cenário a probabilidade já está **abaixo de 46%** no modelo — não faz sentido "
                                            "«subir» IEG ou IDA só para passar no número. Use os controlos para ver o que manteria "
                                            "o aluno alinhado ao que a escola observa."
                                        )
                                    elif _pres.status == "found" and _pres.ieg is not None and _pres.ida is not None:
                                        st.success(
                                            "Para ficar **abaixo de 46%** neste modelo (com os outros eixos fixos), uma combinação "
                                            "possível seria **IEG** em "
                                            f"**{_pres.ieg:.2f}** e **IDA** em **{_pres.ida:.2f}** "
                                            "(entre os pares testados, é uma das que exige **menos** alteração a partir do cenário atual)."
                                        )
                                    else:
                                        st.warning(
                                            "Não foi possível descer **abaixo de 46%** só subindo **IEG** e **IDA** até **10** "
                                            "(passo 0,25). Outros eixos (atitude, trajetória, contexto) também pesam no modelo — "
                                            "veja o **gráfico de fatores** (SHAP) e ajuste outros controlos, ou converse com a equipe."
                                        )
        
                                st.markdown(
                                    _risk_subheading_muted(
                                        "Comparar ficha e simulação",
                                        "A **ficha** é o registo na base; a **simulação** usa os valores dos controlos.",
                                    ),
                                    unsafe_allow_html=True,
                                )
                                cmp1, cmp2 = st.columns((1, 1), gap="small")
                                with cmp1:
                                    st.metric(
                                        label="Risco na ficha (base)",
                                        value=f"{proba * 100:.1f}%",
                                        help="Probabilidade com os dados originais do relatório.",
                                    )
                                with cmp2:
                                    st.metric(
                                        label="Risco com este cenário",
                                        value=f"{float(new_p) * 100:.1f}%",
                                        help="Probabilidade depois de alterar os controlos; a diferença em relação à ficha está no quadro abaixo.",
                                    )
        
                                if abs(delta_pp) < 0.08:
                                    if sim_matches_base_ficha(sim, _sim_baseline):
                                        _delta_html = (
                                            "<strong>Igual à ficha:</strong> os controlos reproduzem o registo na base — "
                                            "a percentagem coincide. <strong>Altere os indicadores</strong> para explorar outros cenários."
                                        )
                                    else:
                                        _delta_html = (
                                            "<strong>Quase igual à ficha:</strong> a diferença é menor que **0,1** na escala de 0 a 100 — "
                                            "as mudanças nos controlos pouco mexeram no resultado para este aluno."
                                        )
                                elif delta_pp > 0:
                                    _delta_html = (
                                        f"<strong>Em relação à ficha, o risco subiu cerca de {delta_pp:.1f} pontos</strong> "
                                        "na escala de 0 a 100 — o modelo lê <strong>mais</strong> probabilidade de defasagem ou evasão."
                                    )
                                else:
                                    _delta_html = (
                                        f"<strong>Em relação à ficha, o risco desceu cerca de {abs(delta_pp):.1f} pontos</strong> "
                                        "na escala de 0 a 100 — o modelo lê <strong>menos</strong> probabilidade de defasagem ou evasão."
                                    )
                                st.markdown(
                                    f'<div class="pm-risk-delta-box">{_delta_html}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(
                                    "Em documentos técnicos esta diferença chama-se **pontos percentuais (p.p.)**: "
                                    "cada ponto = 1 unidade na escala de 0 a 100% da probabilidade."
                                )
        
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
                "**Próximo passo:** indique um **RA** válido ou escolha um **nome** na lista acima para ver risco na ficha, "
                "SHAP e parecer do Theo (e, se precisar, simulação no modo técnico)."
            )

    with tab2:
        st.markdown(
            _risk_heading_html(
                "Lista ordenada por risco",
                "Filtros com **fases 1–8** e **turmas A–E** normalizados (remove duplicados por grafia). "
                "A tabela mostra os alunos com maior probabilidade estimada.",
                tight_top=True,
            ),
            unsafe_allow_html=True,
        )
        st.info(
            "Ordenação automática pelo **modelo** sobre os dados carregados — não substitui listas oficiais nem decisões "
            "da escola."
        )
        fases_opts = ranking_fase_options(df)
        turmas_opts = ranking_turma_letter_options(df)
        turmas_ui = [""] + turmas_opts
        if not fases_opts:
            st.warning("Não há valores de **Fase** reconhecíveis (1–8) na base carregada.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                f_sel = st.selectbox(
                    "Fase do programa",
                    fases_opts,
                    index=0,
                    format_func=lambda x: f"Fase {int(x)}",
                    key="pm_rsk_tab2_fase",
                )
            with c2:
                t_sel = st.selectbox(
                    "Turma (opcional)",
                    turmas_ui,
                    index=0,
                    format_func=lambda x: "Todas as turmas" if x == "" else f"Turma {x}",
                    key="pm_rsk_tab2_turma",
                )
            with c3:
                topn = st.selectbox("Quantos no topo", [10, 20], index=0, key="pm_rsk_tab2_top")
            mask = ranking_mask(df, int(f_sel), str(t_sel))
            ranked = predict_risk_batch(bundle, df, mask).head(topn)
            if ranked.empty:
                st.warning(
                    "**Nenhum aluno** neste filtro — experimente outra **fase**, escolha **Todas as turmas** ou verifique se a base "
                    "tem registros para essa combinação."
                )
            else:
                show = ranked.copy()
                show["Prob. risco (%)"] = (show["risco"] * 100).round(1)
                out = show[["RA", "Nome", "Fase", "Turma", "Ano", "Prob. risco (%)"]].copy()
                st.dataframe(
                    out,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Prob. risco (%)": st.column_config.ProgressColumn(
                            "Prob. risco",
                            help="Probabilidade estimada pelo modelo (0–100%). Barras mais cheias = maior valor.",
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                        ),
                        "Nome": st.column_config.TextColumn("Nome", width="medium"),
                    },
                )


def render_dictionary(df: pd.DataFrame) -> None:
    st.subheader("Dicionário de dados")
    st.caption("Edite as descrições e clique em **Salvar dicionário**; o Theo usa este texto no SQL e nas respostas.")
    rows = load_dictionary()
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

    with st.expander("Pré-visualizar dados (Parquet / CSV)", expanded=False):
        st.markdown(
            "Confira valores reais ao lado das descrições do dicionário. "
            "**Parquet** = o mesmo ficheiro e normalização usados no chat e nos dashboards. "
            "**CSV** = `data/relatorio.csv`, se existir (útil para confrontar com exportações)."
        )
        st.caption(f"Parquet em uso: `{pq_show}`")
        preview_source = st.radio(
            "Origem",
            ("Parquet (dados carregados)", "CSV (relatorio.csv)"),
            horizontal=True,
            key="dict_preview_source",
        )
        n_preview = st.slider("Linhas a mostrar", 25, 500, 100, 25, key="dict_preview_n")

        if preview_source.startswith("Parquet"):
            view = df.head(n_preview)
            st.dataframe(view, width="stretch", hide_index=True)
            st.caption(f"Total na base carregada: **{len(df)}** linhas × {len(df.columns)} colunas (mostrando {min(n_preview, len(df))}).")
            sample_csv = view.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descarregar amostra (CSV)",
                data=sample_csv,
                file_name="amostra_parquet_carregado.csv",
                mime="text/csv",
                key="dict_dl_parquet_sample",
            )
        elif not csv_path.exists():
            st.info(f"Não há ficheiro em `{csv_show}`. Coloque um export CSV aí para pré-visualizar.")
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
                total_hint = (
                    f"Mostrando as primeiras **{len(csv_df)}** linhas lidas do ficheiro"
                    + (" (limite do controlo acima)." if len(csv_df) < n_preview else ".")
                )
                st.caption(total_hint)
                st.download_button(
                    "Descarregar esta pré-visualização (CSV)",
                    data=csv_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="amostra_relatorio_csv.csv",
                    mime="text/csv",
                    key="dict_dl_csv_sample",
                )


def _embed_annual_pdf(path: Path, pdf_bytes: bytes) -> None:
    """Leitor PDF integrado no Streamlit (`st.pdf`); recua para iframe só se necessário."""
    _pdf_h = min(960, max(520, int(900 * 0.88)))
    pdf_fn = getattr(st, "pdf", None)
    if pdf_fn is not None:
        try:
            pdf_fn(str(path.resolve()), height=_pdf_h)
            return
        except Exception:
            try:
                pdf_fn(pdf_bytes, height=_pdf_h)
                return
            except Exception as exc:
                st.warning(
                    f"O leitor PDF nativo falhou ({exc}). "
                    "Instale o componente: `pip install streamlit-pdf` (ou `pip install -r requirements.txt`) e **reinicie** a app."
                )
    b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")
    if len(b64) > 6_000_000:
        st.warning(
            "O PDF é grande demais para pré-visualização embutida neste navegador. Use o descarregar acima."
        )
        return
    iframe = (
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="100%" '
        f'style="min-height:{_pdf_h}px;border:none;" title="Relatório anual"></iframe>'
    )
    components.html(iframe, height=_pdf_h + 32, scrolling=True)


def render_annual_report() -> None:
    st.title("Relatório anual")
    st.caption(
        "O documento abre **nesta página**, com o leitor do Streamlit. O ficheiro deve estar em "
        "**assets/relatorio_anual.pdf**. Opcional: link **Gamma** (variável **PM_RELATORIO_ANUAL_GAMMA_URL** ou secret "
        "**RELATORIO_ANUAL_GAMMA_URL**)."
    )
    pdf_bytes: bytes | None = None
    if not _ANNUAL_PDF_PATH.exists():
        st.warning(
            f"PDF não encontrado: `{_ANNUAL_PDF_PATH.relative_to(_ROOT)}`. "
            "Coloque o relatório nesse caminho para o ver aqui e para descarregar."
        )
    else:
        try:
            pdf_bytes = _ANNUAL_PDF_PATH.read_bytes()
        except OSError as exc:
            st.error(f"Não foi possível ler o PDF: {exc}")

    url = _annual_gamma_url()
    row_a, row_b = st.columns([2, 1], gap="small")
    with row_a:
        if url:
            st.link_button("Abrir versão interativa no Gamma.app", url, use_container_width=True)
        else:
            st.info(
                "Sem URL do Gamma: defina **PM_RELATORIO_ANUAL_GAMMA_URL** ou **RELATORIO_ANUAL_GAMMA_URL** em secrets."
            )
    with row_b:
        if pdf_bytes is not None:
            st.download_button(
                label="Descarregar PDF",
                data=pdf_bytes,
                file_name=_ANNUAL_PDF_PATH.name,
                mime="application/pdf",
                key="annual_pdf_dl",
            )

    if pdf_bytes is None:
        return

    st.divider()
    st.subheader("Documento")
    _embed_annual_pdf(_ANNUAL_PDF_PATH, pdf_bytes)


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
    dict_rows = load_dictionary()
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
