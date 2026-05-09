from __future__ import annotations

from passos_magico.llm.sql_parse import sql_guard_disallowed_tokens


def test_guard_blocks_feedback_when_not_in_columns() -> None:
    cols = ["ra", "nome", "ano", "inde"]
    sql = "SELECT feedback FROM dados WHERE ano = 2022"
    ok, err = sql_guard_disallowed_tokens(sql, cols)
    assert not ok
    assert "feedback" in err.lower()


def test_guard_allows_feedback_when_column_exists() -> None:
    cols = ["feedback", "ano"]
    sql = "SELECT feedback FROM dados LIMIT 5"
    assert sql_guard_disallowed_tokens(sql, cols) == (True, "")


def test_guard_ignores_feedback_inside_string_literal() -> None:
    cols = ["ano"]
    sql = "SELECT ano FROM dados WHERE nome = 'tem feedback no texto' LIMIT 5"
    assert sql_guard_disallowed_tokens(sql, cols) == (True, "")


def test_guard_skips_when_no_known_columns() -> None:
    assert sql_guard_disallowed_tokens("SELECT 1 FROM dados", None) == (True, "")
    assert sql_guard_disallowed_tokens("SELECT 1 FROM dados", []) == (True, "")
