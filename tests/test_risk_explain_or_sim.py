"""Bullets de interpretação do simulador e comparação cenário vs ficha na base."""

from __future__ import annotations

from passos_magico.ui.risk_sim_copy import risk_explain_lines, sim_matches_base_ficha, snapshot_sim_baseline


def test_explain_above_threshold_without_pedagogical_rules() -> None:
    lines = risk_explain_lines(0.68, 3, 9.0, 8.0, 8.0)
    assert any("46%" in x for x in lines)


def test_explain_below_zone_low_no_rules() -> None:
    lines = risk_explain_lines(0.28, 3, 9.0, 8.0, 8.0)
    assert any("30%" in x or "abaixo" in x.lower() for x in lines)


def test_explain_zone_mid_no_rules() -> None:
    lines = risk_explain_lines(0.40, 3, 9.0, 8.0, 8.0)
    assert any("30%" in x and "46%" in x for x in lines)


def test_explain_quartzo_adds_line() -> None:
    lines = risk_explain_lines(0.30, 1, 9.0, 8.0, 8.0)
    assert any("Quartzo" in x for x in lines)


def test_explain_ieg_low() -> None:
    lines = risk_explain_lines(0.30, 3, 7.0, 8.0, 8.0)
    assert any("engajamento" in x.lower() for x in lines)


def test_explain_choque() -> None:
    lines = risk_explain_lines(0.30, 3, 9.0, 10.0, 8.0)
    assert any("Choque" in x or "choque" in x.lower() for x in lines)


def test_sim_matches_base_baseline_roundtrip() -> None:
    sim = {
        "Fase": 8.0,
        "Turma_ord": 4.0,
        "Ano": 2024.0,
        "INDE": 7.0,
        "IDA": 8.0,
        "IAN": 9.0,
        "IEG": 8.0,
        "IPV": 7.0,
        "Pedra_ord": 3.0,
        "IAA": 7.5,
        "IPS": 7.0,
        "IPP": 7.0,
        "ING": 7.0,
        "MAT": 7.0,
        "POR": 8.0,
        "Delta_INDE": 0.0,
    }
    base = snapshot_sim_baseline(sim)
    assert sim_matches_base_ficha(sim, base)
    sim2 = dict(sim)
    sim2["INDE"] = 8.0
    assert not sim_matches_base_ficha(sim2, base)


def test_sim_matches_float_tolerance() -> None:
    sim = {
        "Fase": 8.0,
        "Turma_ord": 4.0,
        "Ano": 2024.0,
        "INDE": 7.0,
        "IDA": 8.0,
        "IAN": 9.0,
        "IEG": 8.0,
        "IPV": 7.0,
        "Pedra_ord": 3.0,
        "IAA": 7.5,
        "IPS": 7.0,
        "IPP": 7.0,
        "ING": 7.0,
        "MAT": 7.0,
        "POR": 8.0,
        "Delta_INDE": 0.0,
    }
    base = snapshot_sim_baseline(sim)
    sim2 = dict(sim)
    sim2["INDE"] = 7.02
    assert sim_matches_base_ficha(sim2, base)
