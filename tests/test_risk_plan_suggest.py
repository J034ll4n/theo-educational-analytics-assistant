"""Sugestão mínima IEG/IDA para risco abaixo do limiar."""

from __future__ import annotations

from passos_magico.ui.risk_plan_suggest import suggest_minimal_ieg_ida


def test_already_below_threshold() -> None:
    def pred(_sim: dict) -> float:
        return 0.30

    r = suggest_minimal_ieg_ida({"IEG": 7.0, "IDA": 7.0, "k": 1}, pred, threshold=0.46)
    assert r.status == "already_below"
    assert r.ieg is None and r.ida is None


def test_impossible_nan_initial() -> None:
    def pred(_sim: dict) -> float:
        return float("nan")

    r = suggest_minimal_ieg_ida({"IEG": 5.0, "IDA": 5.0}, pred)
    assert r.status == "impossible"


def test_impossible_never_drops() -> None:
    def pred(_sim: dict) -> float:
        return 0.99

    r = suggest_minimal_ieg_ida({"IEG": 9.0, "IDA": 9.0}, pred, step=0.25)
    assert r.status == "impossible"


def test_found_prefers_smaller_quadratic_cost() -> None:
    """(6,5) e (5,6) ambos válidos com custo 1 — desempate lexicográfico (5,6)."""

    def pred(sim: dict) -> float:
        ieg = float(sim["IEG"])
        ida = float(sim["IDA"])
        if ieg >= 6.0 and ida >= 5.0:
            return 0.30
        if ieg >= 5.0 and ida >= 6.0:
            return 0.30
        return 0.80

    r = suggest_minimal_ieg_ida({"IEG": 5.0, "IDA": 5.0}, pred, threshold=0.46, step=1.0)
    assert r.status == "found"
    assert r.ieg == 5.0 and r.ida == 6.0


def test_found_not_first_lex_pair() -> None:
    """Primeiro na grelha que passa pode não ser o de menor custo."""

    def pred(sim: dict) -> float:
        ieg = float(sim["IEG"])
        ida = float(sim["IDA"])
        if ieg >= 8.0 and ida >= 5.0:
            return 0.30
        if ieg >= 5.0 and ida >= 8.0:
            return 0.30
        return 0.80

    r = suggest_minimal_ieg_ida({"IEG": 5.0, "IDA": 5.0}, pred, threshold=0.46, step=1.0)
    assert r.status == "found"
    # (8,5) custo 9, (5,8) custo 9 — empate lex (5,8) < (8,5)
    assert r.ieg == 5.0 and r.ida == 8.0
