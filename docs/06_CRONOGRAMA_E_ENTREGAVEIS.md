# Cronograma de desenvolvimento e entregáveis (Fase 5)

A construção foi organizada em etapas sequenciais, com entregas parciais testáveis. Os nomes das fases abaixo são uma conveniência de documentação; ajuste as datas conforme o calendário oficial da sua turma na FIAP.

## Fase A — Fundação de dados

- Definição do esquema esperado do relatório (colunas de indicadores e identificadores).
- Implementação do ETL CSV → Parquet com normalização de tipos.
- Criação do dicionário de dados inicial e arquivo JSON versionável.

**Entregável**: `data/dados.parquet` reproduzível a partir de `relatorio.csv`; documentação das colunas.

## Fase B — Motor analítico e interface

- Integração DuckDB com Parquet e validação de SQL somente leitura.
- Protótipo Streamlit com carregamento dos dados e visualização básica.
- Definição da navegação por áreas funcionais (chat, risco, dicionário).

**Entregável**: aplicação executável localmente sem LLM (consultas poderiam ser manuais neste estágio intermediário).

## Fase C — Assistente Theo (LLM local)

- Configuração do Ollama e escolha do modelo open-source adequado ao hardware disponível.
- Prompts de sistema para geração de SQL, insights e sugestões; parsing do SQL na resposta.
- Pipeline SQL → tabela → gráfico Plotly; exportação de imagem.

**Entregável**: chat analítico funcional com dados reais ou sintéticos.

## Fase D — Machine learning e explicabilidade

- Definição documentada do rótulo de risco e engenharia de features.
- Treino Random Forest, persistência em `modelo.joblib`, relatório de métricas no console.
- Integração na UI: probabilidade, SHAP, simulação e triagem em lote.
- Recalculo da coluna `risco` no ETL após existência do modelo.

**Entregável**: aba de previsão alinhada ao Parquet; risco materializado opcionalmente na base.

## Fase E — Consolidação da entrega Postech

- Script `run.bat` para onboarding (venv, dependências, ETL, treino condicional, Streamlit).
- Revisão de privacidade (processamento local) e checklist de teste manual.
- Pacote de documentação em `docs/` e README na raiz.

**Entregável**: repositório ou pacote zip com instruções, código e dados de exemplo; vídeo ou apresentação conforme exigência da disciplina (não incluso neste repositório por padrão).

## Artefatos esperados na avaliação típica

| Artefato | Local sugerido |
|----------|------------------|
| Código-fonte | Raiz do projeto + `passos_magico/` |
| Dados de exemplo | `data/relatorio.csv` |
| Modelo treinado | `models/modelo.joblib` (gerado) |
| Documentação | `docs/` + `README.md` |
| Dependências | `requirements.txt` |
