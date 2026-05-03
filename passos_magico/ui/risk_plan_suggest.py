"""Sugestão de ajuste IEG/IDA para reduzir risco simulado — lógica pura, testável."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np


@dataclass(frozen=True)
class IegIdaPlanResult:
    """Resultado de `suggest_minimal_ieg_ida`."""

    status: Literal["already_below", "found", "impossible"]
    ieg: float | None = None
    ida: float | None = None


def suggest_minimal_ieg_ida(
    sim_template: dict[str, Any],
    predict_fn: Callable[[dict[str, Any]], float],
    *,
    threshold: float = 0.46,
    step: float = 0.25,
    max_value: float = 10.0,
) -> IegIdaPlanResult:
    """
    Procura o par (IEG, IDA) com menor «custo» (distância quadrática ao cenário atual)
    tal que P(risco) < threshold, incrementando a partir dos valores atuais até max_value.

    `predict_fn` recebe uma cópia mutável do cenário com IEG/IDA atualizados.
    """
    sim: dict[str, Any] = dict(sim_template)
    p0 = float(predict_fn(sim))
    if p0 != p0:  # NaN
        return IegIdaPlanResult(status="impossible")
    if p0 < float(threshold):
        return IegIdaPlanResult(status="already_below")

    start_ieg = float(sim["IEG"])
    start_ida = float(sim["IDA"])
    best: tuple[float, float] | None = None
    best_cost = float("inf")

    for ieg_v in np.arange(start_ieg, max_value + step * 0.5, step):
        for ida_v in np.arange(start_ida, max_value + step * 0.5, step):
            sim["IEG"] = float(round(float(ieg_v), 2))
            sim["IDA"] = float(round(float(ida_v), 2))
            p = float(predict_fn(sim))
            if p != p or not (p < float(threshold)):
                continue
            cost = (sim["IEG"] - start_ieg) ** 2 + (sim["IDA"] - start_ida) ** 2
            candidate = (sim["IEG"], sim["IDA"])
            if cost < best_cost - 1e-12:
                best_cost = cost
                best = candidate
            elif abs(cost - best_cost) <= 1e-12 and best is not None:
                if candidate < best:
                    best = candidate

    if best is None:
        return IegIdaPlanResult(status="impossible")
    return IegIdaPlanResult(status="found", ieg=best[0], ida=best[1])
