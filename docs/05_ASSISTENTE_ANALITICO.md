# Assistente analítico (Theo + LLM local)

## Papel do assistente

O módulo converte perguntas em linguagem natural em:

1. Uma consulta **SQL** adequada ao esquema conhecido (tabela lógica `dados`).
2. Um **conjunto de resultados** (DataFrame) para tabela e gráfico.
3. Texto de **insight** e **sugestões** de continuidade da conversa.

Tudo isso depende do serviço **Ollama** estar em execução na máquina e do modelo configurado em `passos_magico/llm/config.py` estar baixado (`ollama pull <nome>`).

## Segurança das consultas (`passos_magico/data_engine/query.py`)

- Apenas instruções que começam com `SELECT` são aceitas.
- Palavras-chave de escrita ou DDL (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.) são bloqueadas.
- Uma única instrução por vez; múltiplos comandos separados por `;` são rejeitados.
- Se a consulta não trouxer `LIMIT`, o sistema acrescenta **LIMIT 5000** para proteção de memória.

Isso reduz risco de alteração acidental dos dados e mantém o comportamento previsível em ambiente acadêmico.

## Motor SQL

DuckDB cria uma view em memória:

```sql
CREATE VIEW dados AS SELECT * FROM read_parquet('<caminho>');
```

Assim o modelo pode sempre referenciar `dados` nas respostas, independentemente do caminho físico do arquivo.

## Gráficos

O pipeline gera figura Plotly a partir do DataFrame resultante. A interface permite **alterar o tipo de gráfico** quando os dados tabulares estão disponíveis, e oferece download em PNG (Kaleido).

## Dicionário como contexto

O conteúdo editável em `dicionario.json` (e na aba correspondente) é convertido em bloco de texto para o prompt. Isso melhora a aderência do SQL ao vocabulário institucional (nomes de colunas, significado de indicadores).

## Comportamento quando o LLM está indisponível

Se o Ollama não responder na URL configurada, o chat informa indisponibilidade e não executa o fluxo completo — a aba de **machine learning** continua utilizável com o modelo tabular já treinado.

## Dependências de integração

- **LangChain** (`ChatOllama`): abstrai chamadas ao endpoint local do Ollama.
- **requests**: verificação rápida de disponibilidade (`/api/tags`).
