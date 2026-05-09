"""Parecer do Theo sobre os dashboards: gera texto via Ollama e cache em ficheiro até os dados mudarem."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from passos_magico.data_engine.loader import PROJECT_ROOT, get_parquet_path
from passos_magico.llm.ollama_client import invoke_string, ollama_available
from passos_magico.ml.risk_display import OPERATIONAL_HIGH_RISK_THRESHOLD
from passos_magico.ui.dashboard_catalog import IAN_ADEQUACAO, INDICATOR_CATALOG

FEEDBACK_PATH: Path = PROJECT_ROOT / "data" / "dashboard_theo_feedback.txt"
META_PREFIX = "DASHBOARD_FEEDBACK_META_JSON:"
# Incluir no fingerprint para invalidar cache quando mudarem regras do parecer.
FEEDBACK_PROMPT_VERSION = "v4-voz-conselheira-panorama"

DASHBOARD_FEEDBACK_SYSTEM = """És o **Theo**, analista de dados educacionais da Passos Mágicos.
Recebes um resumo **quantitativo** extraído dos **mesmos** dados que alimentam os dashboards (KPIs, Panorama e aba Indicador).
O leitor é equipa pedagógica / direção: quer **clareza profissional** sem frieza — como um **colega experiente** que partilha leitura e ideias ao lado do relatório, **sem** substituir decisões da escola.

**Língua e tom:** português europeu; frases completas; **cordial e directo** («convém», «vale a pena notar», «se calhar»), **nunca** condescendente. Evita buzzwords vazios («sinergias», «holístico») e frases de relatório genérico.

**Regras de facto:**
- Não inventes números que não apareçam no resumo.
- Não afirmes que «viste» gráficos interactivos; baseias-te só no resumo.
- Siglas correctas: **IEG** (engajamento), nunca «EIG».
- Máximo ~1200 palavras.

**Proibido (molde de má qualidade):** repetir em cada indicador a mesma estrutura do tipo «A média global do X é Y com desvio Z. A tendência por ano mostra uma melhoria moderada…». Isso **cansa** e parece texto de modelo — **varia** aberturas, comprimento e ângulo (comparação entre anos, contraste com outro indicador do resumo, uma pergunta curta ocasional, um aviso quando os números **oscilam** entre anos).

**Cobertura obrigatória:** na secção «## Leitura dos indicadores e tendências», um `###` por indicador **nesta ordem**, cada um com **pelo menos uma frase com números** do resumo (média, média por ano, % acima de limiar no risco, etc.):
### INDE
### IDA
### IAN
### IEG
### IPV
### IPS
### IPP
### risco

Se o resumo disser ausência ou poucos dados: «**SIGLA**: não disponível ou dados insuficientes no resumo.» — mantém o `###`.

**Conteúdo por secção:**
- **## Panorama geral** — 3 a 6 frases: período, escala (alunos / linhas se constar), **uma** ideia-força que ligue o conjunto (ex.: desempenho global vs. volatilidade em algum eixo). Tom de abertura acolhedor, não lista de KPIs.
- **## Leitura dos indicadores e tendências** — por cada `###`, **1 a 4 frases**: números concretos; se `média por Ano` estiver no resumo, comenta **subidas, descidas ou saltos** de forma honesta (incluindo anos mais fracos). O **IPS** e o **risco** merecem leitura prudente, sem alarmismo nem complacência.
- **## Risco e priorização** — integra média de probabilidade **e**, se existir no resumo, **fracção acima do limiar operacional**; explica em linguagem humana o que isso **pode** significar para triagem (sempre com ressalva de contexto escolar).
- **## Recomendações** — lista **numerada** (4 a 7 itens), **concretos** e ligados aos **achados** anteriores (não «continuar a monitorizar» sozinho como item único). Mistura prioridades pedagógicas e dados que realmente destacaste.

**Estrutura markdown obrigatória (títulos exactos):**
## Panorama geral
## Leitura dos indicadores e tendências
## Risco e priorização
## Recomendações"""


def compute_dashboard_fingerprint(df: pd.DataFrame) -> str:
    """Identifica a versão dos dados: Parquet no disco + agregados estáveis do DataFrame."""
    parts: list[Any] = [FEEDBACK_PROMPT_VERSION, int(len(df)), sorted(str(c) for c in df.columns)]
    p = get_parquet_path()
    if p.exists():
        stt = p.stat()
        parts.extend([int(stt.st_mtime_ns), int(stt.st_size)])
    num_cols = sorted(
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or c == "risco"
    )[:40]
    agg: dict[str, tuple[float, int]] = {}
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        agg[c] = (float(s.sum(skipna=True)), int(s.count()))
    parts.append(agg)
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_meta_and_body() -> tuple[dict[str, Any] | None, str]:
    if not FEEDBACK_PATH.exists():
        return None, ""
    text = FEEDBACK_PATH.read_text(encoding="utf-8")
    if not text.startswith(META_PREFIX):
        return None, text.strip()
    nl = text.find("\n\n")
    if nl == -1:
        return None, text
    meta_line = text[:nl]
    body = text[nl + 2 :].strip()
    try:
        meta = json.loads(meta_line[len(META_PREFIX) :])
    except json.JSONDecodeError:
        return None, body
    return meta, body


def load_cached_feedback(fingerprint: str) -> str | None:
    meta, body = _read_meta_and_body()
    if meta is None or meta.get("fingerprint") != fingerprint:
        return None
    return body if body else None


def save_feedback(fingerprint: str, body: str) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "fingerprint": fingerprint,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out = META_PREFIX + json.dumps(meta, ensure_ascii=False) + "\n\n" + body.strip() + "\n"
    FEEDBACK_PATH.write_text(out, encoding="utf-8")


def build_dashboard_summary_for_llm(df: pd.DataFrame) -> str:
    """Texto compacto com estatísticas de **todos** os painéis (panorama + indicadores do catálogo)."""
    d = df
    lines: list[str] = []

    n = len(d)
    n_ra = int(d["RA"].nunique()) if "RA" in d.columns else n
    lines.append(f"Linhas: {n} | Alunos únicos (RA): {n_ra}")
    if "Ano" in d.columns:
        ay = pd.to_numeric(d["Ano"], errors="coerce")
        lines.append(f"Anos: {float(ay.min())} – {float(ay.max())} (registos por ano: {ay.value_counts().sort_index().to_dict()})")

    if "INDE" in d.columns:
        x = pd.to_numeric(d["INDE"], errors="coerce")
        lines.append(f"INDE: média={x.mean():.3f} desvio={x.std():.3f} min={x.min():.3f} max={x.max():.3f}")
    if "IDA" in d.columns:
        x = pd.to_numeric(d["IDA"], errors="coerce")
        lines.append(f"IDA: média={x.mean():.3f} desvio={x.std():.3f}")
    if "IAN" in d.columns:
        x = pd.to_numeric(d["IAN"], errors="coerce")
        lines.append(
            f"IAN: média={x.mean():.3f} | fração ≥ {IAN_ADEQUACAO}: {(x >= IAN_ADEQUACAO).mean():.1%}"
        )
    for code, label in (
        ("IEG", "Engajamento"),
        ("IPV", "Ponto de virada"),
        ("IPS", "Psicossocial"),
        ("IPP", "Pedagógico complementar"),
    ):
        if code not in d.columns:
            lines.append(f"{code} ({label}): coluna ausente no dataset.")
            continue
        x = pd.to_numeric(d[code], errors="coerce")
        if x.notna().sum() < 3:
            lines.append(f"{code} ({label}): dados insuficientes (poucos valores válidos).")
            continue
        lines.append(
            f"{code} ({label}): média={x.mean():.4f} std={x.std():.4f} min={x.min():.4f} max={x.max():.4f}"
        )
    if "risco" not in d.columns:
        lines.append("risco (prob. alto risco, modelo ML): coluna ausente no dataset.")
    else:
        r = pd.to_numeric(d["risco"], errors="coerce")
        if r.notna().sum() < 3:
            lines.append("risco (prob. alto risco, modelo ML): dados insuficientes ou todos ausentes.")
        else:
            lim = OPERATIONAL_HIGH_RISK_THRESHOLD
            lines.append(
                f"risco (prob. alto risco, modelo ML): média={r.mean():.3f} | P≥{lim:.2f}={(r >= lim).mean():.1%} | "
                f"quartis Q1={r.quantile(0.25):.3f} mediana={r.median():.3f} Q3={r.quantile(0.75):.3f}"
            )

    if "IAN" in d.columns and "Ano" in d.columns:
        g = (
            d.assign(y=pd.to_numeric(d["IAN"], errors="coerce"), a=pd.to_numeric(d["Ano"], errors="coerce"))
            .dropna(subset=["y", "a"])
            .groupby("a")["y"]
            .agg(["mean", "count"])
        )
        lines.append("IAN médio por Ano: " + g.to_dict()["mean"].__repr__())

    if "Ano" in d.columns and "RA" in d.columns:
        c = d.groupby(pd.to_numeric(d["Ano"], errors="coerce"))["RA"].nunique()
        lines.append("Alunos únicos por Ano: " + c.to_dict().__repr__())

    if "INDE" in d.columns and "Pedra" in d.columns:
        g = d.groupby("Pedra", observed=True)["INDE"].mean()
        lines.append("INDE médio por Pedra: " + g.to_dict().__repr__())

    cols_corr = [c for c in ("INDE", "IDA", "IAN", "IEG", "IPV", "IPS", "IPP") if c in d.columns]
    if len(cols_corr) >= 2:
        num = d[cols_corr].apply(pd.to_numeric, errors="coerce")
        cmat = num.corr()
        lines.append("Matriz de correlação (Pearson) entre colunas numéricas disponíveis:")
        lines.append(cmat.round(3).to_string())

    lines.append("\n--- Por indicador (média global; aba Indicador do dashboard) ---")
    for code, title in INDICATOR_CATALOG:
        if code not in d.columns:
            continue
        s = pd.to_numeric(d[code], errors="coerce")
        if s.notna().sum() < 5:
            continue
        lines.append(
            f"{code} ({title}): média={s.mean():.4f} std={s.std():.4f} min={s.min():.4f} max={s.max():.4f}"
        )
        if "Fase" in d.columns:
            byf = d.assign(_v=s).dropna(subset=["_v", "Fase"]).groupby("Fase", observed=True)["_v"].mean()
            lines.append(f"  média por Fase (top 5): {byf.head(5).to_dict()}")
        if "Ano" in d.columns:
            bya = (
                d.assign(_v=s, a=pd.to_numeric(d["Ano"], errors="coerce"))
                .dropna(subset=["_v", "a"])
                .groupby("a")["_v"]
                .mean()
            )
            lines.append(f"  média por Ano: {bya.to_dict()}")

    lines.append(
        "\n--- CHECKLIST PARA O MODELO (não copiar literalmente para o utilizador) ---"
    )
    lines.append(
        "O texto final DEVE usar `### INDE`, `### IDA`, … `### risco` nesta ordem; variar redacção entre blocos; "
        "sigla **IEG** correcta."
    )

    return "\n".join(lines)


def render_dashboard_theo_feedback(df: pd.DataFrame, *, compact: bool = False) -> None:
    """Parecer em cache ou geração via Ollama. `compact=True` omite título longo (ex.: dentro de expander)."""
    if not compact:
        st.markdown("### Parecer do Theo sobre os dashboards")
        st.caption(
            "Leitura automática dos **mesmos números** dos gráficos (Panorama + indicadores), em tom de **conversa com a equipa** — "
            f"não substitui o vosso juízo. Texto em cache em `{FEEDBACK_PATH.relative_to(PROJECT_ROOT)}`; "
            "regenera quando o Parquet ou os agregados mudam (ou quando actualizamos as regras de redacção)."
        )

    fp = compute_dashboard_fingerprint(df)
    cached = load_cached_feedback(fp)

    if cached:
        st.markdown(cached)
        st.caption("Em cache para esta versão dos dados.")
        return

    if not ollama_available():
        st.info(
            "**Ollama** não está acessível a partir desta instância — o parecer automático não foi gerado. "
            "Os **gráficos e KPIs** da página funcionam na mesma. Em **local**, inicie o Ollama para gerar texto; "
            f"também pode colocar texto fixo em `{FEEDBACK_PATH.relative_to(PROJECT_ROOT)}` com o meta JSON correcto."
        )
        return

    with st.spinner("O Theo está a ler o resumo dos dashboards e a redigir recomendações…"):
        try:
            summary = build_dashboard_summary_for_llm(df)
            user = (
                "Segue o resumo quantitativo unificado (panorama + indicadores). "
                "Redige o parecer com a **voz conselheira** do sistema: profissional, calor humano, **sem** repetir o mesmo "
                "molde frase a frase em cada indicador. Inclui **os oito ###** na ordem indicada.\n\n"
                f"{summary}"
            )
            text = invoke_string(DASHBOARD_FEEDBACK_SYSTEM, user, temperature=0.12)
        except ImportError as exc:
            st.error(
                "Faltam pacotes para o LLM (ex.: **langchain-community**, **langchain-core**). "
                f"Instale-os no ambiente ou use só os gráficos sem parecer automático. Detalhe: {exc}"
            )
            return
        except Exception as exc:
            st.error(f"Não foi possível gerar o parecer: {exc}")
            return

    save_feedback(fp, text)
    st.success("Parecer gerado e guardado.")
    st.markdown(text)
