"""Texto de diagnóstico Theo para o painel de ML."""

from __future__ import annotations

from passos_magico.llm.ollama_client import invoke_string, ollama_available
from passos_magico.llm.prompts import ML_DIAGNOSIS_SYSTEM


def build_ml_context(
    nome: str,
    ra: str,
    proba: float,
    shap_pairs: list[tuple[str, float]],
) -> str:
    lines = [
        f"Aluno: {nome} (RA {ra}).",
        f"Probabilidade estimada de risco de defasagem/evasão: {proba * 100:.1f}%.",
        "Impacto aproximado por variável (SHAP ou importância):",
    ]
    for name, val in shap_pairs[:8]:
        lines.append(f"- {name}: {val:+.4f}")
    return "\n".join(lines)


def generate_diagnosis_text(
    nome: str,
    ra: str,
    proba: float,
    shap_pairs: list[tuple[str, float]],
) -> str:
    if not ollama_available():
        return (
            f"**Theo:** {nome} apresenta cerca de **{proba * 100:.0f}%** de probabilidade de risco. "
            "Principais fatores no modelo: "
            + ", ".join(f"{n}" for n, _ in shap_pairs[:3])
            + ". (Conecte o Ollama para um parecer mais detalhado.)"
        )
    ctx = build_ml_context(nome, ra, proba, shap_pairs)
    return invoke_string(
        ML_DIAGNOSIS_SYSTEM,
        ctx,
        temperature=0.15,
    )
