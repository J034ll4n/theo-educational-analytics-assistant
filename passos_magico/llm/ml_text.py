"""Texto de diagnóstico Theo para o painel de ML."""

from __future__ import annotations

from typing import Any

from passos_magico.llm.ficha_quality_hint import ficha_quality_snapshot_extra_lines
from passos_magico.llm.ollama_client import invoke_string, ollama_available
from passos_magico.llm.prompts import ML_DIAGNOSIS_SYSTEM


def _ficha_snapshot_block(feats: dict[str, Any] | None) -> str:
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
        return "\n".join(lines) + "\n"
    except (TypeError, ValueError):
        return ""


def build_ml_context(
    nome: str,
    ra: str,
    proba: float,
    shap_pairs: list[tuple[str, float]],
    feats: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"Aluno: {nome} (RA {ra}).",
        f"Probabilidade estimada de risco de defasagem/evasão: {proba * 100:.1f}%.",
        "Impacto aproximado por variável (SHAP ou importância):",
    ]
    for name, val in shap_pairs[:8]:
        lines.append(f"- {name}: {val:+.4f}")
    lines.append("")
    lines.append(
        "Lembrete do gráfico SHAP neste painel: barra à direita da origem (zero) = tende a **aumentar** o risco no "
        "modelo; à esquerda = tende a **reduzir**."
    )
    snap = _ficha_snapshot_block(feats)
    if snap:
        lines.append("")
        lines.append(snap.rstrip())
    return "\n".join(lines)


def generate_diagnosis_text(
    nome: str,
    ra: str,
    proba: float,
    shap_pairs: list[tuple[str, float]],
    theo_context_block: str = "",
    feats: dict[str, Any] | None = None,
) -> str:
    if not ollama_available():
        return (
            f"**Theo:** {nome} apresenta cerca de **{proba * 100:.0f}%** de probabilidade de risco. "
            "Principais fatores no modelo: "
            + ", ".join(f"{n}" for n, _ in shap_pairs[:3])
            + ". (Conecte o Ollama para um parecer mais detalhado.)"
        )
    ctx = build_ml_context(nome, ra, proba, shap_pairs, feats=feats)
    if theo_context_block.strip():
        ctx = f"{theo_context_block.strip()}\n\n---\n{ctx}"
    return invoke_string(
        ML_DIAGNOSIS_SYSTEM,
        ctx,
        temperature=0.15,
    ).strip()
