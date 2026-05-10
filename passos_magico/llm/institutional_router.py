"""Heurística para perguntas respondíveis só com texto do resumo anual (sem SQL em `dados`)."""

from __future__ import annotations

import re

# Perguntas que pedem narrativa do relatório/resumo (inclui menções a "pdf" por hábito do usuário)
_NARRATIVE_PATTERNS = (
    # «feedback» / «feedbacks» (plural) + relatório (qualquer ordem)
    r"\bfeedbacks?\b.*\b(relat[oó]rio\s+anual|gamma|resumo\s+anual|pede)\b",
    r"\b(relat[oó]rio\s+anual|gamma|resumo\s+anual|pede)\b.*\bfeedbacks?\b",
    r"\b(parecer|opini[aã]o|avalia[cç][aã]o|impress[aã]o|cr[ií]tica)\b.*\b(relat[oó]rio\s+anual|gamma|resumo\s+anual|pede)\b",
    r"\bs[ií]ntese\s+institucional\b",
    r"\b(narrativa|storytelling)\b.*\b(relat[oó]rio\s+anual|gamma|resumo\s+anual|pede)\b",
    r"\b(o\s+que\s+diz|o\s+que\s+fala)\b.*\b(gamma|relat[oó]rio\s+anual|resumo\s+anual)\b",
    r"\b(texto|conte[uú]do)\s+do\s+(gamma|relat[oó]rio\s+anual|resumo\s+anual)\b",
    r"\bresumo\s+anual\b",
    r"\b(no|do|da|em)\s+resumo\s+anual\b",
    r"\bpdf\b",
    r"\bgamma\b",
    r"\bno\s+site\b.*\b(relat[oó]rio|passos|datathon)\b",
    r"\brelat[oó]rio\b.*\bpede\b",
    r"\bpede\b.*\b20\d{2}\b",
    r"\bobjetivo[s]?\b.*\b(relat[oó]rio|pede|pdf|resumo)\b",
    r"\b(resuma|resumo|síntese|sintese|descreva|explique)\b.*\b(pede|pdf|relat[oó]rio|resumo\s+anual)\b",
    r"\b(conteúdo|conteudo)\b.*\b(pdf|relat[oó]rio|resumo)\b",
    r"\bsegundo\s+o\s+(pdf|relat[oó]rio|resumo)\b",
    r"\bno\s+texto\s+do\s+(pdf|relat[oó]rio|resumo)\b",
    r"\binsights?\b.*\b(pede|pdf|relat[oó]rio|resumo\s+anual)\b",
)

# Se aparecerem, a pergunta provavelmente exige dados tabulares (não só texto institucional)
_DATA_STRONG_PATTERNS = (
    r"\bm[eé]dia[s]?\b",
    r"\b(m[eé]dia|total|soma)\b.*\b(relat[oó]rio\s+anual|gamma|resumo\s+anual|pede)\b",
    r"\bquantos\b",
    r"\bquantas\b",
    r"\bcontagem\b",
    r"\bpor\s+turma\b",
    r"\bpor\s+fase\b",
    r"\bdistribui",
    r"\bevolu[cç][aã]o\b",
    r"\bgr[aá]fico\b",
    r"\bindicador\b.*\b(ida|inde|ian|ieg|ipv)\b",
    r"\btop\s*\d",
    r"\branking\b",
    r"\bcompar\w+\s+(o\s+)?inde\b",
    r"\bselect\b",
    r"\btabela\s+dados\b",
    r"\brisco\b",
    r"\bshap\b",
)


def has_annual_block_in_context(theo_context_block: str) -> bool:
    tb = theo_context_block or ""
    return "### Resumo anual institucional" in tb or "### Contexto narrativo (Gamma / relatório anual)" in tb


def is_institutional_narrative_only(question: str, theo_context_block: str) -> bool:
    """
    True se a pergunta deve ser respondida só com o bloco institucional (resumo em texto),
    sem consulta SQL à tabela `dados`.
    """
    if not has_annual_block_in_context(theo_context_block):
        return False
    q = question.strip().lower()
    if len(q) < 8:
        return False

    if any(re.search(p, q, re.IGNORECASE) for p in _DATA_STRONG_PATTERNS):
        return False

    if any(re.search(p, q, re.IGNORECASE) for p in _NARRATIVE_PATTERNS):
        return True

    # Perguntas muito curtas só sobre "o pede" / "o relatório" sem pedir números
    if re.search(r"^(o\s+)?que\s+[ée]\s+o\s+pede", q):
        return True
    if "objetivo" in q and ("pede" in q or "relat" in q):
        return True

    return False
