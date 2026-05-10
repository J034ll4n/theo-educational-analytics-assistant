from __future__ import annotations

from passos_magico.llm.institutional_router import is_institutional_narrative_only

_CTX = """
### Resumo anual institucional (PEDE_PASSOS)
Exemplo de texto do relatório.
"""

_CTX_GAMMA = """
### Contexto narrativo (Gamma / relatório anual)
Texto de exemplo do site Gamma.
"""


def test_feedback_sobre_relatorio_anual_is_narrative_only() -> None:
    assert is_institutional_narrative_only(
        "me dê um feedback sobre o relatório anual",
        _CTX,
    )
    assert is_institutional_narrative_only(
        "mê de um feedback sobre o relatório anual",
        _CTX,
    )


def test_feedbacks_plural_relatorio_anual_ordem_natural() -> None:
    """«feedbacks» (plural) + relatório antes ou depois — deve ir ao texto institucional, não SQL com todos os RA."""
    assert is_institutional_narrative_only(
        "com base no nosso relatório anual, quais feedbacks você pode me trazer?",
        _CTX,
    )
    assert is_institutional_narrative_only(
        "quais feedbacks você traz com base no relatório anual?",
        _CTX_GAMMA,
    )


def test_quantos_still_uses_sql_not_narrative_only() -> None:
    assert not is_institutional_narrative_only(
        "quantos alunos aparecem no resumo anual em números?",
        _CTX,
    )


def test_no_annual_block_never_narrative_only() -> None:
    assert not is_institutional_narrative_only(
        "me dê um feedback sobre o relatório anual",
        "só dicionário, sem bloco anual",
    )


def test_sintese_institucional_narrative() -> None:
    assert is_institutional_narrative_only(
        "faça uma síntese institucional do que está no PEDE",
        _CTX,
    )


def test_o_que_diz_o_gamma_with_gamma_block() -> None:
    assert is_institutional_narrative_only(
        "o que diz o gamma sobre as prioridades da ONG?",
        _CTX_GAMMA,
    )


def test_narrativa_relatorio_anual() -> None:
    assert is_institutional_narrative_only(
        "qual é a narrativa do relatório anual para apresentar na reunião?",
        _CTX,
    )


def test_media_no_relatorio_forca_sql() -> None:
    assert not is_institutional_narrative_only(
        "qual a média de INDE no relatório anual em números?",
        _CTX,
    )
