"""Testes leves para contexto do parecer ML (Theo)."""

from __future__ import annotations

from passos_magico.llm.shap_labels import shap_feature_label_pt


def test_shap_feature_label_pt_pipeline_names() -> None:
    assert "IAN" in shap_feature_label_pt("ian")
    assert "amplitude" in shap_feature_label_pt("range_inde").lower()
    assert "IAN" in shap_feature_label_pt("num__ian")


def test_shap_feature_label_pt_legacy_order() -> None:
    assert "Fase" in shap_feature_label_pt("Fase")
