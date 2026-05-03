"""Limiares de risco exibidos na UI — alinhados ao treino (Youden / notebook ML)."""

# Probabilidade mínima para classificar "alto risco operacional" em KPIs e dashboards.
OPERATIONAL_HIGH_RISK_THRESHOLD: float = 0.46

# Limite inferior da faixa "atenção moderada" no texto do simulador (abaixo do operacional).
MODERATE_ATTENTION_LOWER: float = 0.30
