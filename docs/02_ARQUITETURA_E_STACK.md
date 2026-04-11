# Arquitetura e stack tecnológica

## Visão em camadas

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit (app/main.py) — UI, sessão, downloads      │
├─────────────────────────────────────────────────────────┤
│  passos_magico/                                         │
│    · llm/     — prompts, Ollama, pipeline SQL→gráfico   │
│    · data_engine/ — DuckDB em memória sobre Parquet    │
│    · ml/      — features, inferência, SHAP             │
│    · semantic/— dicionário JSON para contexto ao LLM    │
│    · ui/      — estilos e helpers de interface         │
├─────────────────────────────────────────────────────────┤
│  Dados: data/dados.parquet  ·  Modelo: models/*.joblib │
└─────────────────────────────────────────────────────────┘
         ↕ local
   Ollama (LLM) · Python 3.10+
```

## Tecnologias principais

| Camada | Tecnologia | Papel |
|--------|------------|--------|
| Interface | Streamlit | Aplicação web local de uma só página com navegação entre abas funcionais |
| Consultas | DuckDB | Motor SQL analítico; lê Parquet via view `dados` em memória |
| Gráficos | Plotly (+ Kaleido para export PNG) | Visualizações interativas e download de imagem |
| LLM | Ollama + LangChain (`ChatOllama`) | Geração de SQL e texto explicativo em ambiente local |
| Machine learning | scikit-learn (RandomForestClassifier), joblib | Classificação binária de risco; persistência do modelo |
| Explicabilidade | SHAP | Contribuição de cada feature na predição individual |
| Dados tabulares | pandas, pyarrow | Leitura/escrita Parquet e manipulação |

## Fluxo resumido de uma pergunta no chat

1. O usuário formula a pergunta; o sistema carrega o bloco de texto do **dicionário de dados** como contexto semântico.
2. O modelo local gera uma instrução SQL; o código valida que é apenas `SELECT` e executa sobre o Parquet.
3. O resultado vira tabela e gráfico (com heurística de tipo e opção de troca manual).
4. O modelo gera insight em texto e sugestões de próximas perguntas.

## Decisões de projeto

- **Parquet**: compacto e tipado; adequado a colunas numéricas dos indicadores.
- **DuckDB**: SQL familiar para o LLM; isolamento por processo sem servidor de banco separado.
- **Ollama**: reprodutibilidade em laboratório e em máquinas sem GPU dedicada, desde que o modelo caiba na RAM/VRAM disponível.

## Estrutura de pastas (referência)

- `app/` — ponto de entrada da aplicação.
- `passos_magico/` — biblioteca interna com domínio (dados, ML, LLM, UI).
- `scripts/` — ETL e treinamento offline.
- `data/` — `relatorio.csv` (entrada) e `dados.parquet` (processado).
- `models/` — artefatos `modelo.joblib` gerados pelo treino.
