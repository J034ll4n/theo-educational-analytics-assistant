from __future__ import annotations

from passos_magico.llm.error_messages import humanize_sql_execution_error


def test_humanize_referenced_column() -> None:
    msg = 'Binder Error: Referenced column "feedback" not found in FROM clause!'
    out = humanize_sql_execution_error(msg)
    assert "feedback" in out
    assert "não existe" in out.lower() or "existem" in out.lower()


def test_humanize_group_by_aggregate() -> None:
    msg = "Binder Error: GROUP BY clause cannot contain aggregates!"
    out = humanize_sql_execution_error(msg)
    assert "agrupamento" in out.lower() or "agrupa" in out.lower()


def test_humanize_empty_message() -> None:
    assert "reformular" in humanize_sql_execution_error("").lower()
