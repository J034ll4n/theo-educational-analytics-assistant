"""Texto de diagnóstico Theo para o painel de ML."""

from __future__ import annotations

from typing import Any

import pandas as pd

from passos_magico.llm.ficha_quality_hint import ficha_quality_snapshot_extra_lines
from passos_magico.llm.ml_scenarios import build_scenarios_markdown, format_inde_history_summary
from passos_magico.llm.ollama_client import invoke_string, ollama_available
from passos_magico.llm.prompts import ML_DIAGNOSIS_SYSTEM
from passos_magico.llm.shap_labels import shap_feature_label_pt


def _ficha_snapshot_block(feats: dict[str, Any] | None, eng_row: pd.Series | None = None) -> str:
    if not feats:
        return ""
    try:
        fase = int(feats.get("Fase", 0))
        ano = int(feats.get("Ano", 0))
        turma_ord = float(feats.get("Turma_ord", 0))
        pedra_n = int(round(float(feats.get("Pedra_ord", 0))))
        _pedra_names = {1: "Quartzo", 2: "Ágata", 3: "Ametista", 4: "Topázio"}
        pedra_l = _pedra_names.get(pedra_n, str(pedra_n))
        lines = [
            "Valores principais na ficha (relatório / base — não simulados):",
            f"- Fase {fase}, ano {ano}, turma (ord.) {turma_ord:.0f}, pedra {pedra_l}",
            f"- INDE {float(feats.get('INDE', 0)):.1f}, IDA {float(feats.get('IDA', 0)):.1f}, IAN {float(feats.get('IAN', 0)):.1f}",
            f"- IEG {float(feats.get('IEG', 0)):.1f}, IPV {float(feats.get('IPV', 0)):.1f}",
        ]
        lines.extend(ficha_quality_snapshot_extra_lines(feats))
        if eng_row is not None and "media_turma_inde" in eng_row.index:
            mt = eng_row.get("media_turma_inde")
            if mt is not None and not (isinstance(mt, float) and pd.isna(mt)):
                lines.append(
                    f"- Média INDE do **grupo** (instituição / fase / ano na base): **{float(mt):.2f}** "
                    f"(comparar com INDE **{float(feats.get('INDE', 0)):.2f}** do aluno neste ano)."
                )
        return "\n".join(lines) + "\n"
    except (TypeError, ValueError):
        return ""


def build_ml_context(
    nome: str,
    ra: str,
    proba: float,
    shap_pairs: list[tuple[str, float]],
    feats: dict[str, Any] | None = None,
    *,
    eng_row: pd.Series | None = None,
    inde_history_line: str = "",
) -> str:
    lines = [
        f"Aluno: {nome} (RA {ra}).",
        f"Probabilidade estimada de risco de defasagem/evasão: {proba * 100:.1f}%.",
        "Impacto aproximado por fator (SHAP): na tua resposta usa **sempre o nome legível** da coluna abaixo, "
        "nunca só o identificador técnico entre parênteses.",
    ]
    for name, val in shap_pairs[:10]:
        leg = shap_feature_label_pt(name)
        lines.append(f"- **{leg}** (técnico: `{name}`): {val:+.4f}")
    lines.append("")
    lines.append(
        "Leitura do gráfico SHAP: valor **positivo** → este fator, neste modelo, **empurra o risco para cima**; "
        "valor **negativo** → **empurra para baixo**. Não confundir com «nota boa ou má» fora do modelo."
    )
    snap = _ficha_snapshot_block(feats, eng_row=eng_row)
    if snap:
        lines.append("")
        lines.append(snap.rstrip())
    if inde_history_line.strip():
        lines.append("")
        lines.append(inde_history_line.strip())
    return "\n".join(lines)


def generate_diagnosis_text(
    nome: str,
    ra: str,
    proba: float,
    shap_pairs: list[tuple[str, float]],
    theo_context_block: str = "",
    feats: dict[str, Any] | None = None,
    *,
    bundle: Any | None = None,
    df: pd.DataFrame | None = None,
    eng_row: pd.Series | None = None,
) -> str:
    inde_hist = ""
    scen_block = ""
    if feats and df is not None and not df.empty:
        inde_hist = format_inde_history_summary(df, ra)
    if feats and bundle is not None and df is not None and not df.empty:
        scen_block = build_scenarios_markdown(bundle, df, ra, feats, eng_row, proba)

    if not ollama_available():
        base = (
            f"**Theo:** {nome} apresenta cerca de **{proba * 100:.0f}%** de probabilidade de risco. "
            "Fatores que mais pesam no modelo (leitura SHAP): "
            + ", ".join(shap_feature_label_pt(n) for n, _ in shap_pairs[:3])
            + ". (Conecte o Ollama para um parecer mais detalhado.)"
        )
        extra = ""
        if scen_block:
            extra = "\n\n" + scen_block
        return base + extra

    student = build_ml_context(
        nome,
        ra,
        proba,
        shap_pairs,
        feats=feats,
        eng_row=eng_row,
        inde_history_line=inde_hist,
    ).strip()
    if scen_block:
        student = student + "\n\n" + scen_block
    head = (
        "### Dados do caso individual (única fonte para percentagem de risco, SHAP e conclusões sobre ESTE aluno)\n\n"
        + student
    )
    if theo_context_block.strip():
        ctx = head + "\n\n---\n\n" + theo_context_block.strip()
    else:
        ctx = head
    return invoke_string(
        ML_DIAGNOSIS_SYSTEM,
        ctx,
        temperature=0.22,
    ).strip()
