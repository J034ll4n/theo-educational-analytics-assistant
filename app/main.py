"""Entrada Streamlit — Theo, painel de risco e dicionário."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Garante imports `app` e `passos_magico` quando o CWD não é a raiz do projeto
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.io as pio
from plotly.io import to_image

from app.cached import cached_load_dados, cached_model_bundle, cached_sql_result
from passos_magico.llm.ml_text import generate_diagnosis_text
from passos_magico.llm.ollama_client import ollama_available
from passos_magico.llm.pipeline import (
    sql_and_chart_step,
    stream_insight_text,
    suggestions_step,
)
from passos_magico.ml.features import FEATURE_ORDER, row_features_from_df
from passos_magico.ml.inference import explain_row_shap, predict_risk_batch, predict_row_features
from passos_magico.llm.charts import CHART_TYPE_OPTIONS, figure_from_dataframe, heuristic_kind_to_chart_id
from passos_magico.semantic.metadata import load_dictionary, rows_to_prompt_block, save_dictionary
from passos_magico.ui.sidebar_helper import render_open_sidebar_button
from passos_magico.ui.styles import inject_global_css

# Páginas: id interno, rótulo curto no menu, descrição para a lateral
PAGE_DEFS: list[tuple[str, str, str]] = [
    (
        "chat",
        "💬  Chat analítico",
        "Converse com o Theo: ele gera SQL nos seus dados (DuckDB), mostra um gráfico e explica o resultado. "
        "Use perguntas sobre médias, turmas, anos ou indicadores (IDA, INDE, etc.). Tudo roda localmente com Ollama.",
    ),
    (
        "risk",
        "🎯  Previsão de risco",
        "Modelo de ML treinado nos indicadores do relatório estima probabilidade de risco escolar por aluno. "
        "Inclui explicação SHAP, simulador what-if e triagem em lote por fase/turma.",
    ),
    (
        "dict",
        "📖  Dicionário de dados",
        "Edite descrições das colunas: o Theo usa esse texto como contexto ao gerar SQL e respostas. "
        "Salve para aplicar na próxima pergunta do chat.",
    ),
]
PAGE_LABELS: dict[str, str] = {p[0]: p[1] for p in PAGE_DEFS}
PAGE_HELP: dict[str, str] = {p[0]: p[2] for p in PAGE_DEFS}
_MAX_CHART_ROWS = 5000


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
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
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
    if p < 0.35:
        return "#3fb950"
    if p < 0.65:
        return "#d29922"
    return "#f85149"


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


def render_chat(
    dictionary_block: str,
    ollama_ok: bool,
    sql_runner,
) -> None:
    st.subheader("Chat analítico — Theo")
    st.caption(
        "O Theo gera SQL nos seus dados, exibe um gráfico e explica o resultado. "
        "Abaixo do gráfico você pode trocar o **tipo de visualização** (barras, linhas, pizza, etc.) "
        "sem refazer a pergunta — útil quando o gráfico automático não ficou ideal."
    )

    # Dentro de um container, o Streamlit usa chat_input em modo inline (logo abaixo do texto),
    # em vez de fixar a barra no rodapé da janela.
    with st.container():
        prompt = st.chat_input(
            "Faça uma pergunta sobre os dados educacionais…",
            key="pm_chat_input",
        )
    pending = st.session_state.pop("pending_prompt", None)
    user_text = (pending or prompt or "").strip()

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})

    for idx, msg in enumerate(st.session_state.messages):
        _render_one_chat_message(msg, idx)

    msgs = st.session_state.messages
    if not msgs or msgs[-1]["role"] != "user":
        return

    user_text = msgs[-1]["content"]
    msg_id = len(msgs)

    with st.chat_message("assistant", avatar="🎓"):
        try:
            _run_assistant_turn(
                user_text,
                dictionary_block,
                ollama_ok,
                sql_runner,
                msg_id,
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
    dictionary_block: str,
    ollama_ok: bool,
    sql_runner,
    msg_id: int,
) -> None:
    with st.spinner("Buscando dados…"):
        step = sql_and_chart_step(
            user_text,
            dictionary_block,
            ollama_ok,
            sql_executor=sql_runner,
        )
    if step.sql_error or step.df is None:
        err = step.sql_error or "Erro desconhecido."
        text = (
            f"**Theo:** Não consegui concluir a análise. Detalhes: {err}"
            if ollama_ok
            else "**Theo:** Ollama indisponível. Inicie o serviço local e tente novamente."
        )
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
            stream_insight_text(user_text, dictionary_block, step.df, step.chart_kind)
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


def render_risk(df: pd.DataFrame, bundle: dict[str, Any]) -> None:
    st.subheader("Previsão de risco escolar")
    st.caption(
        "O modelo usa indicadores do relatório para estimar probabilidade de risco por aluno. "
        "SHAP mostra o que mais influenciou; o simulador altera variáveis para ver o efeito; "
        "a aba em lote ranqueia alunos por fase/turma."
    )
    tab1, tab2 = st.tabs(["Perfil e simulador", "Triagem em lote"])

    with tab1:
        names = df["Nome"].astype(str) + " — RA " + df["RA"].astype(str)
        ra_from_select = None
        c1, c2 = st.columns(2)
        with c1:
            ra_input = st.text_input("Registro do aluno (RA)", placeholder="RA2020001")
        with c2:
            pick = st.selectbox("Ou selecione o aluno", options=[""] + list(names))
        if pick:
            ra_from_select = str(df.loc[names == pick, "RA"].iloc[0])
        ra = (ra_from_select or ra_input or "").strip()
        feats = row_features_from_df(df, ra) if ra else None

        if feats:
            base_feats = {k: feats[k] for k in FEATURE_ORDER}
            proba = predict_row_features(bundle, base_feats)
            shap_pairs = explain_row_shap(bundle, base_feats)

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Prob. de risco",
                f"{proba * 100:.1f}%",
                help="Probabilidade estimada de alto risco (defasagem/evasão).",
            )
            m2.metric("Fase", f"{int(feats['Fase'])}", delta_color="normal")
            m3.metric("Ano", f"{int(feats['Ano'])}", delta_color="normal")

            st.markdown(
                f"<div style='padding:12px;border-radius:8px;background:#21262d;border:1px solid #30363d;'>"
                f"<span style='color:{risk_color(proba)};font-size:2rem;font-weight:700;'>"
                f"{proba * 100:.0f}%</span> — score de risco (interpretação: "
                "verde abaixo de 35%, amarelo entre 35% e 65%, vermelho acima de 65%)</div>",
                unsafe_allow_html=True,
            )

            labels = [p[0] for p in shap_pairs[:12]]
            vals = [p[1] for p in shap_pairs[:12]]
            fig_bar = go.Figure(
                go.Bar(
                    x=vals,
                    y=labels,
                    orientation="h",
                    marker_color="#58a6ff",
                )
            )
            fig_bar.update_layout(
                template="plotly_dark",
                title="Impacto por variável (SHAP / importância)",
                margin=dict(l=120, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_bar, width="stretch")

            left, right = st.columns(2)
            with left:
                st.markdown("##### Simulador What-If")
                sim: dict[str, float] = dict(base_feats)
                sim["Fase"] = st.slider("Fase", 1.0, 8.0, float(sim["Fase"]), 1.0)
                sim["Turma_ord"] = st.slider("Turma (1=A … 4=D)", 1.0, 5.0, float(sim["Turma_ord"]), 1.0)
                sim["Ano"] = st.slider("Ano", float(df["Ano"].min()), float(df["Ano"].max()), float(sim["Ano"]), 1.0)
                sim["INDE"] = st.slider("INDE", 0.0, 10.0, float(sim["INDE"]), 0.1)
                sim["IDA"] = st.slider("IDA", 0.0, 10.0, float(sim["IDA"]), 0.1)
                sim["IAN"] = st.slider("IAN", 0.0, 10.0, float(sim["IAN"]), 0.1)
                sim["IEG"] = st.slider("IEG", 0.0, 10.0, float(sim["IEG"]), 0.1)
                sim["IPV"] = st.slider("IPV", 0.0, 10.0, float(sim["IPV"]), 0.1)
                sim["Pedra_ord"] = st.slider("Pedra (1–4)", 1.0, 4.0, float(sim["Pedra_ord"]), 1.0)
                new_p = predict_row_features(bundle, sim)
                st.metric(
                    "Risco simulado",
                    f"{new_p * 100:.1f}%",
                    delta=f"{(new_p - proba) * 100:+.1f} p.p.",
                    delta_color="inverse",
                )

            with right:
                st.markdown("##### Diagnóstico (Theo + ML)")
                with st.container(border=True):
                    diag = generate_diagnosis_text(feats.get("Nome", ""), ra, proba, shap_pairs)
                    st.markdown(diag)
        else:
            st.info("Informe um RA válido ou escolha um nome na lista.")

    with tab2:
        fases = sorted(df["Fase"].dropna().unique().tolist())
        turmas = [""] + sorted(df["Turma"].dropna().astype(str).unique().tolist())
        c1, c2, c3 = st.columns(3)
        with c1:
            f_sel = st.selectbox("Fase", fases, index=0)
        with c2:
            t_sel = st.selectbox("Turma (opcional)", turmas)
        with c3:
            topn = st.selectbox("Top N", [10, 20], index=0)
        mask = df["Fase"] == f_sel
        if t_sel:
            mask = mask & (df["Turma"].astype(str) == t_sel)
        ranked = predict_risk_batch(bundle, df, mask).head(topn)
        st.dataframe(ranked, width="stretch", hide_index=True)


def render_dictionary() -> None:
    st.subheader("Dicionário de dados")
    st.caption(
        "Descreva cada coluna em linguagem natural: esse texto vira contexto para o Theo ao gerar SQL e respostas. "
        "Clique em **Salvar dicionário** para aplicar na próxima mensagem do chat."
    )
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


def main() -> None:
    st.set_page_config(
        page_title="Passos Mágicos — Painel local",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sidebar_page" not in st.session_state:
        st.session_state.sidebar_page = "chat"

    # Reabre a sidebar (fallback ao ícone do topo)
    render_open_sidebar_button()

    ollama_ok = ollama_available()
    df = cached_load_dados()
    bundle = cached_model_bundle()
    dict_rows = load_dictionary()
    dictionary_block = rows_to_prompt_block(dict_rows)

    with st.sidebar:
        st.markdown(
            '<p class="pm-branded-kicker">PASSOS MÁGICOS · PAINEL LOCAL</p>'
            '<p class="pm-branded-slogan">A <span class="pm-hero-magia">Magia</span> da '
            '<span class="pm-hero-inov">Inovação</span>, '
            '<span class="pm-hero-passo">Passo a Passo</span>.</p>'
            '<p class="pm-sidebar-hint">Menu fechou? Use o botão <strong>☰ Menu</strong> no topo do conteúdo '
            "ou o ícone no <strong>canto superior esquerdo</strong>.</p>",
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

    sql_runner = cached_sql_result

    if page == "chat":
        render_chat(dictionary_block, ollama_ok, sql_runner)
    elif page == "risk":
        render_risk(df, bundle)
    else:
        render_dictionary()


if __name__ == "__main__":
    main()
