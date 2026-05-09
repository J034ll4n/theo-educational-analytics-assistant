"""Catálogo de indicadores partilhado entre dashboards e feedback do Theo."""

INDICATOR_CATALOG: list[tuple[str, str]] = [
    ("INDE", "Índice de Desenvolvimento Educacional"),
    ("IDA", "Desempenho académico (aprendizagem)"),
    ("IAN", "Adequação ao nível"),
    ("IEG", "Engajamento"),
    ("IPV", "Ponto de virada"),
    ("IPS", "Índice psicossocial"),
    ("IPP", "Índice pedagógico complementar"),
    ("risco", "Probabilidade de alto risco (modelo ML)"),
]

IAN_ADEQUACAO: float = 6.5
