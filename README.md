# Assistente Theo — Passos Mágicos (Postech FIAP · Fase 5)

Aplicação local (Streamlit) para apoio à gestão educacional: assistente analítico com **LLM via Ollama** (localhost), consultas sobre **DuckDB/Parquet** e previsão de risco com **scikit-learn** (Random Forest + SHAP).

**Documentação completa da concepção e da entrega:** pasta [`docs/`](docs/00_INDICE.md) (índice com visão de negócio, arquitetura, dados, ML, assistente, cronograma e manual de instalação).

## Pré-requisitos

1. **Python 3.10+** com launcher `py` ou `python` no PATH.
2. **Ollama** instalado e em execução: [https://ollama.com](https://ollama.com)
3. Baixar um modelo local, por exemplo:

   ```bash
   ollama pull llama3
   ```

   Ajuste o nome do modelo em `passos_magico/llm/config.py` se usar outro (ex.: `mistral`).

## Como executar

1. Coloque `data/relatorio.csv` na pasta `data` (o repositório inclui um exemplo sintético).
2. Dê duplo clique em **`run.bat`** (ou execute no terminal na raiz do projeto).

O script cria `.venv`, instala dependências, gera `data/dados.parquet` e `models/modelo.joblib` se ainda não existirem e inicia o Streamlit.

## Privacidade

Nenhum dado de alunos é enviado à nuvem: LLM e inferência rodam apenas na máquina local.

## Checklist rápido (QA)

- O serviço **Ollama** está em execução e o modelo configurado foi baixado (`ollama pull`).
- `data/dados.parquet` e `models/modelo.joblib` existem (gerados pelo `run.bat` na primeira execução).
- Com a rede desligada, a interface e o modelo ML ainda funcionam; apenas o Ollama precisa estar disponível **localmente**.

## Estrutura

- `app/main.py` — entrada Streamlit e navegação.
- `passos_magico/` — camadas de dados, LLM, ML e UI.
- `scripts/etl.py` — CSV → Parquet.
- `scripts/train_model.py` — treina e salva o modelo de risco.
