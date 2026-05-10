"""Bloco temporal no pedido SQL do Theo."""

from passos_magico.llm.prompts import build_sql_user_message


def test_build_sql_user_message_includes_operational_year_block():
    msg = build_sql_user_message(
        "Quantos alunos temos?",
        "### Dicionário\n(colunas…)",
        dados_columns=None,
        reference_year_default=2024,
        reference_years_span=(2020, 2024),
    )
    assert "Ano de referência operacional (chat)" in msg
    assert "WHERE Ano = 2024" in msg
    assert "COUNT(DISTINCT RA)" in msg
    assert "2020" in msg and "2024" in msg
