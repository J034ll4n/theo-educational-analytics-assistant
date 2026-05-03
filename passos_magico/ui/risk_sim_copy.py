"""Textos e helpers do simulador de risco (sem Streamlit)."""

from __future__ import annotations

from typing import Any

from passos_magico.ml.features import FEATURE_ORDER
from passos_magico.ml.risk_display import (
    MODERATE_ATTENTION_LOWER,
    OPERATIONAL_HIGH_RISK_THRESHOLD,
)

_SIM_COMPARE_KEYS: tuple[str, ...] = tuple(FEATURE_ORDER) + ("IAA", "IPS", "MAT", "POR", "Delta_INDE")


def snapshot_sim_baseline(sim: dict[str, Any]) -> dict[str, float]:
    """Valores da ficha na base usados nos sliders (para comparar cenário atual vs baseline)."""
    return {k: float(sim[k]) for k in _SIM_COMPARE_KEYS if k in sim}


def sim_matches_base_ficha(sim: dict[str, Any], baseline: dict[str, float], *, tol: float = 0.051) -> bool:
    """True se os controles coincidem com a ficha na base (tolerância para floats dos sliders)."""
    for k, bv in baseline.items():
        if k not in sim:
            return False
        if abs(float(sim[k]) - float(bv)) > tol:
            return False
    return True


def risk_explain_lines(
    p_sim: float,
    pedra_ord: float,
    ieg: float,
    iaa: float,
    ida: float,
    *,
    threshold: float = OPERATIONAL_HIGH_RISK_THRESHOLD,
    zone_low: float = MODERATE_ATTENTION_LOWER,
) -> list[str]:
    """Bullets em linguagem simples; detalhe técnico só quando ajuda."""
    lines: list[str] = []
    ps = float(p_sim)
    pct = ps * 100.0
    if ps > threshold:
        lines.append(
            f"Neste cenário a estimativa passa dos **{threshold * 100:.0f}%** (limite de atenção usado neste painel). "
            "Vale olhar com calma o **gráfico de fatores** acima (o que mais puxa o risco neste aluno) e o **parecer em texto**, "
            "sempre **junto da equipe**."
        )
    if int(round(float(pedra_ord))) == 1:
        lines.append(
            "A marca **Pedra Quartzo** no histórico continua associada a **mais atenção** neste tipo de modelo."
        )
    if ieg < 8:
        lines.append(
            "O indicador de **engajamento (IEG)** está mais baixo — costuma pesar no acompanhamento."
        )
    if iaa > ida + 1.5:
        lines.append(
            "A **autoavaliação** do aluno está bem mais alta que a nota de **aprendizagem (IDA)**. "
            "Pode haver desalinhamento entre o que ele acha que sabe e o que o relatório mede — "
            "um bom tema para conversa com a equipe (às vezes chamado *choque de realidade*)."
        )
    if not lines:
        if ps > zone_low:
            lines.append(
                f"A estimativa fica entre **cerca de {zone_low * 100:.0f}% e {threshold * 100:.0f}%** "
                f"(hoje **{pct:.0f}%**) — **atenção moderada**. Cruze com outros dados e com a equipe."
            )
        else:
            lines.append(
                f"A estimativa fica **abaixo de cerca de {zone_low * 100:.0f}%** neste cenário (hoje **{pct:.0f}%**) — "
                "leitura mais baixa no modelo; continue cruzando com o que a escola observa no dia a dia."
            )
    elif zone_low < ps <= threshold:
        lines.append(
            f"Com **{zone_low * 100:.0f}% a {threshold * 100:.0f}%** a leitura fica na faixa de **atenção moderada** "
            f"(hoje **{pct:.0f}%**); use também o gráfico de fatores e o contexto da sala."
        )
    return lines
