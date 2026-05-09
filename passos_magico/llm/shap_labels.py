"""Rótulos em português para nomes técnicos de variáveis no parecer ML (Theo)."""

from __future__ import annotations

_SHAP_FEATURE_LABEL_PT: dict[str, str] = {
    "idade": "Idade (anos)",
    "iaa": "IAA — autoavaliação",
    "ieg": "IEG — engajamento",
    "ips": "IPS — percepção socioemocional",
    "ida": "IDA — aprendizagem",
    "ipp": "IPP",
    "ipv": "IPV — ponto de virada",
    "ian": "IAN — adequação ao nível",
    "mat": "Matemática (nota componente)",
    "por": "Português (nota componente)",
    "ing": "Inglês (nota componente)",
    "defas": "Indicador de defasagem (histórico)",
    "delta_inde": "Variação do INDE face ao registo anterior",
    "delta_ian": "Variação do IAN",
    "std_inde": "Variabilidade do INDE ao longo do tempo",
    "media_inde": "Média histórica do INDE do aluno",
    "tendencia_inde": "Tendência recente do INDE",
    "queda_acumulada_inde": "Perda acumulada de INDE face ao melhor momento",
    "range_inde": "Amplitude do INDE (melhor vs pior momento no histórico)",
    "distancia_media_turma": "Distância do INDE do aluno à média INDE do grupo/turma",
    "choque_realidade": "Diferença autoavaliação (IAA) vs desempenho (IDA)",
    "esforco_sem_resultado": "Esforço (engajamento) relativamente ao resultado em aprendizagem",
    "mudanca_pedra": "Mudança de nível Pedra entre registos",
    "fase": "Fase no programa",
    "genero": "Género",
    "instituicao_de_ensino": "Instituição de ensino",
    "pedra": "Nível Pedra (Quartzo, Ágata, …)",
    "pedra_ord": "Nível Pedra (codificação do modelo)",
    "inde": "INDE — desenvolvimento educacional",
    "turma_ord": "Turma (posição A–E no modelo)",
    "ano": "Ano de referência",
}


def shap_feature_label_pt(name: str) -> str:
    """Nome amigável para variável SHAP / importância (código interno do pipeline ou legado)."""
    key = str(name).split("__")[-1].strip().lower()
    return _SHAP_FEATURE_LABEL_PT.get(key, key.replace("_", " ").title())
