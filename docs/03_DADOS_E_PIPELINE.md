# Dados e pipeline de ingestão

## Fonte principal

O arquivo esperado é `data/relatorio.csv`, com colunas alinhadas ao domínio educacional do cenário Passos Mágicos:

- Identificação: `RA`, `Nome`
- Contexto escolar: `Fase`, `Turma`, `Ano`
- Indicadores numéricos (tipicamente 0–10): `INDE`, `IDA`, `IAN`, `IEG`, `IPV`
- Atributo categórico: `Pedra` (programa de desenvolvimento associado ao aluno)

Se o CSV não existir na primeira execução do ETL, o script gera um **conjunto sintético** com distribuição controlada (semente fixa) apenas para permitir desenvolvimento e demonstração — substituir por dados reais quando disponíveis.

## ETL (`scripts/etl.py`)

1. **Leitura** do CSV com pandas.
2. **Normalização de tipos**: `Fase` e `Ano` como inteiros; indicadores como float.
3. **Coluna de risco (ML)**: se existir `models/modelo.joblib`, o pipeline recalcula a coluna `risco` (probabilidade) linha a linha e a inclui no Parquet; caso contrário o Parquet é gravado sem essa coluna e o ETL emite aviso no console.
4. **Saída**: `data/dados.parquet` (formato colunar, eficiente para DuckDB).

O ETL remove uma coluna `risco` pré-existente antes de recalcular, evitando duplicidade ao reprocessar.

## Dicionário de dados (`dicionario.json`)

Arquivo JSON com pares `coluna` / `descricao` utilizados para:

- Exibir ajuda na interface na aba “Dicionário de dados”.
- Injetar contexto nas mensagens enviadas ao modelo ao gerar SQL e explicações.

Alterações salvas na interface passam a valer nas próximas interações do chat, permitindo alinhar vocabulário da escola (sinônimos, regras de negócio) sem alterar código.

## Qualidade e limitações

- Dados faltantes em indicadores são tratados na leitura numérica (`errors='coerce'` no ETL); predições dependem de valores válidos após o enriquecimento de features no módulo de ML.
- O limite padrão de linhas retornadas por consulta SQL no motor analítico é aplicado para proteger memória da interface (detalhes no documento do assistente analítico).
