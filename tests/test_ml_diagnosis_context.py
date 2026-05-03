"""Contexto enviado ao Theo no painel de risco."""

from __future__ import annotations

import pytest


def test_build_ml_context_includes_ficha_when_feats() -> None:
    pytest.importorskip("langchain_community")
    from passos_magico.llm.ml_text import build_ml_context

    feats = {
        "Fase": 6.0,
        "Ano": 2023.0,
        "Turma_ord": 2.0,
        "Pedra_ord": 2.0,
        "INDE": 7.5,
        "IDA": 6.0,
        "IAN": 7.0,
        "IEG": 8.0,
        "IPV": 6.5,
    }
    shap = [("IDA", 0.1), ("IEG", -0.05)]
    ctx = build_ml_context("Ana", "RA1", 0.42, shap, feats=feats)
    assert "INDE 7.5" in ctx
    assert "Fase 6" in ctx
    assert "Lembrete do gráfico SHAP" in ctx
    assert "IDA" in ctx


def test_build_ml_context_without_feats() -> None:
    pytest.importorskip("langchain_community")
    from passos_magico.llm.ml_text import build_ml_context

    ctx = build_ml_context("Ana", "RA1", 0.42, [("IDA", 0.1)])
    assert "Valores principais na ficha" not in ctx
    assert "Lembrete do gráfico SHAP" in ctx
