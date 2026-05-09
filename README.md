# Passos Mágicos — Painel analítico e assistente Theo

**Made by Joe Allan Zirn**

**Disciplina / contexto:** projeto académico (Postech FIAP · Fase 5) — transformação de dados educacionais em decisões de apoio pedagógico.

Este repositório consolida o trabalho desenvolvido: uma **aplicação local em Streamlit** que integra **consulta analítica em linguagem natural** (LLM + SQL sobre dados tabulares), **dashboards**, **previsão de risco escolar com machine learning** (incluindo explicabilidade SHAP), **dicionário de dados editável** e **visualização do relatório institucional** publicado na web. Toda a inferência e os dados sensíveis podem permanecer na máquina do utilizador.

---

## 1. Objetivos do projeto

- Disponibilizar uma interface única para explorar indicadores (INDE, IDA, IAN, IEG, IPV, etc.) e contexto do aluno (fase, turma, pedra, instituição).
- Permitir perguntas em português ao **Theo**, que gera **SQL** sobre uma base **DuckDB** alimentada por **Parquet**, devolvendo tabelas, gráficos e narrativa.
- Estimar **probabilidade de alto risco** com um modelo de ML alinhado ao notebook de treino, com **SHAP** e, em modo técnico, **simulação** de cenários.
- Incorporar o **relatório anual** (apresentação Gamma) na aplicação e dar ao Theo **contexto textual** extraído do mesmo conteúdo (ficheiro Markdown em `assets/`), para respostas alinhadas ao relatório sem confundir com colunas da base.
- Garantir **privacidade**: LLM via **Ollama** em localhost; sem envio obrigatório de dados de alunos a APIs externas.

### 1.1 Expectativas sobre o modelo e o assistente (leia antes da demo)

- **Risco (ML):** a probabilidade na interface é uma **estimativa estatística** treinada em dados históricos. **Não** substitui parecer pedagógico, CIEP, regras da escola nem visita à sala. Os limiares (ex.: 46%) são **marcas operacionais** para leitura na app, não verdades absolutas sobre uma criança.
- **Theo (LLM):** pode **alucinar** ou gerar SQL incorrecto; as respostas devem ser **verificadas** (tabela devolvida, ordens de grandeza). Perguntas “institucionais” usam texto de contexto (Gamma/resumo), não a base linha-a-linha.
- **MVP em nuvem:** dashboards e risco podem funcionar **sem** Ollama; o chat Theo e o parecer automático dos dashboards dependem de **Ollama + LangChain** quando quiser LLM local ou cache pré-gerado.

---

## 2. Stack tecnológica

| Camada | Tecnologias |
|--------|-------------|
| Interface | [Streamlit](https://streamlit.io/) |
| Dados | [Pandas](https://pandas.pydata.org/), [PyArrow](https://arrow.apache.org/docs/python/) |
| Motor analítico | [DuckDB](https://duckdb.org/) sobre Parquet |
| Gráficos | [Plotly](https://plotly.com/python/) |
| LLM | [Ollama](https://ollama.com/) + [LangChain](https://python.langchain.com/) (comunidade) |
| ML | [scikit-learn](https://scikit-learn.org/), [imbalanced-learn](https://imbalanced-learn.org/) (SMOTE), [XGBoost](https://xgboost.readthedocs.io/), [SHAP](https://shap.readthedocs.io/) |
| Empacotamento | `venv`, `requirements.txt`, `run.bat` (Windows) |

---

## 3. Arquitetura (visão geral)

```
relatorio.csv  ──►  ETL (scripts/etl.py)  ──►  dados.parquet
                                                    │
notebooks/Ml/  ──►  treino (notebook)   ──►  modelo_risco_aluno.pkl
                                                    │
                                                    ▼
                    app/main.py (Streamlit) ◄── passos_magico/
                         │                      ├── data_engine/ (loader, SQL)
                         │                      ├── llm/ (Theo, prompts, router)
                         │                      └── ml/ (pipeline de risco, inferência)
```

- **`passos_magico/data_engine`:** resolução do caminho do Parquet, normalização de colunas (aliases RA/Nome/Ano…), execução de SQL.
- **`passos_magico/llm`:** orquestração do assistente, prompts, deteção de perguntas “só institucionais”, geração de gráficos.
- **`passos_magico/ml`:** engenharia de atributos espelhada do notebook (`risk_pipeline.py`), carregamento do modelo (`inference.py`), utilidades de UI para risco.
- **`app/main.py`:** páginas, cache de dados e modelo, CSS global.

---

## 4. Dados e pipeline

### Ficheiro de entrada padrão

- **`data/relatorio.csv`** — exportação tabular com colunas em minúsculas (ex.: `ra`, `ano_referencia`, `inde`, `ian`, …). Este é o ficheiro que deve ser atualizado quando chegam novos dados.

### Parquet usado pela aplicação

A função `get_parquet_path()` em `passos_magico/data_engine/loader.py` escolhe, por ordem:

1. Variável de ambiente **`PASSOS_PARQUET`** (caminho absoluto para um `.parquet` alternativo), se definida;
2. **`notebooks/Ml/relatorio_final.parquet`**, se existir (prioridade típica após ETL/notebook);
3. Caso contrário, **`data/dados.parquet`**.

O ETL em **`scripts/etl.py`** lê `data/relatorio.csv`, normaliza (incluindo sentinélas `-999` → ausente), opcionalmente preenche a coluna **`risco`** se existir modelo carregável, e grava **`data/dados.parquet`**.

### Dicionário semântico

- **`dicionario.json`** — descrições por coluna; o Theo usa este texto como contexto ao gerar SQL e respostas. A página **Dicionário de dados** permite editar e guardar.

---

## 5. Modelo de machine learning (risco do aluno)

### Artefacto principal

- **`modelo_risco_aluno.pkl`** na **raiz do projeto** — ficheiro único que a aplicação carrega para probabilidade de alto risco.

### O que “roda” dentro do modelo (estrutura do `.pkl`)

O ficheiro **não** é só um algoritmo isolado: é um **Pipeline** (scikit-learn + imbalanced-learn) com três etapas na ordem:

1. **`pre` (ColumnTransformer)** — imputação por **mediana** e **StandardScaler** nas variáveis numéricas; **OneHotEncoder** (`handle_unknown='ignore'`) nas categóricas (`fase`, `genero`, `instituicao_de_ensino`, `pedra`).
2. **`smote` (SMOTE)** — sobreamostragem sintética da classe minoritária **apenas no conjunto de treino** dentro de cada fold, para reduzir viés de desequilíbrio entre “sem risco” e “com risco”.
3. **`clf` (classificador)** — o vencedor de uma **competição** entre vários algoritmos, escolhido por **validação cruzada** com métrica **F2** (favorece *recall* da classe positiva, adequado quando o risco é raro).

Candidatos tipicamente avaliados no notebook: **Random Forest**, **XGBoost** (`XGBClassifier`), **regressão logística**, **Gradient Boosting** (sklearn) e **Extra Trees**. O classificador com melhor F2 médio na CV é o que fica gravado no passo `clf`.

### Alvo (rótulo) e variáveis

- **Alvo binário** `alvo_risco`: definido por transição **ano seguinte** com o mesmo aluno (`ra`): condição de risco se o **IAN** cair ou se a **queda do INDE** para o ano seguinte for superior ao limiar definido no notebook (evolução desfavorável).
- **Features numéricas** incluem indicadores brutos (idade, IAA, IEG, IPS, IDA, …), derivadas temporais por aluno (`delta_inde`, `std_inde`, `media_inde`, `tendencia_inde`, …) e construtos pedagógicos (`distancia_media_turma`, `choque_realidade`, `esforco_sem_resultado`, `mudanca_pedra`, etc.).
- **Partição treino/teste:** **GroupShuffleSplit** (~80/20) com agrupamento por **`ra`**, para não misturar linhas do mesmo aluno entre treino e teste (reduz *leakage* temporal).

### Escolhas metodológicas (porquê assim)

| Escolha | Motivo |
|--------|--------|
| **Grupos por RA** no *split* e na CV (**GroupKFold**) | O aluno não aparece em treino e teste ao mesmo tempo; métricas refletem generalização a **novos alunos**, não a novas linhas do mesmo histórico. |
| **F2 na pesquisa de hiperparâmetros** | Em risco escolar importa não deixar passar casos positivos; F2 dá mais peso ao *recall* da classe 1 do que F1. |
| **SMOTE dentro do pipeline** | Equilibrar classes sem abandonar o fluxo sklearn; o passo aplica-se só onde o pipeline é treinado. |
| **Vários modelos + RandomizedSearchCV** | Explorar hiperparâmetros com custo computacional controlado (`n_iter` limitado) e escolher o melhor candidato de forma reprodutível (`random_state` fixo). |
| **Análise de limiar no notebook** | Além do corte 0,5 do `predict`, o notebook procura um **limiar de probabilidade** que maximiza F2 no conjunto de teste — útil para interpretação; a **UI** continua a expor sobretudo a **probabilidade** contínua (0–1). |

### Métricas e resultados reportados no treino

Após o treino, o **notebook** imprime (sobre o conjunto de **teste**, ~20% dos alunos):

- **Ranking** dos candidatos pelo melhor **F2** obtido na CV;
- **Acurácia**, **ROC-AUC**, **PR-AUC** (*average precision*), **precisão / *recall* / F1** por classe e médias macro;
- **Matriz de confusão** (TN, FP, FN, TP);
- **`classification_report`** com limiar 0,5 e, em separado, com o **limiar que maximiza F2** no teste (valores de corte são indicativos e dependem do *split*).

Os **números exatos** dependem do CSV, da semente e do *split*; devem ser **copiados da última execução do notebook** para o relatório escrito da disciplina. A título **ilustrativo** (uma execução local sobre o `data/relatorio.csv` de exemplo do repositório), obtiveram-se ordens de grandeza como: **XGBoost** escolhido na competição, **ROC-AUC ≈ 0,91**, **PR-AUC ≈ 0,64**, acurácia com limiar 0,5 na ordem de **0,79–0,89**, com matriz de confusão a mostrar poucos **falsos negativos** (riscos não detetados) e trade-off com **falsos positivos** conforme o limiar — validar sempre com a saída atual do vosso notebook.

### Código partilhado com a inferência na app

- **`passos_magico/ml/risk_pipeline.py`** — replica a engenharia de atributos quando os dados vêm do Parquet (incluindo variantes de grafia em “Pedra”, idade a partir de datas, etc.).
- **`passos_magico/ml/inference.py`** — carrega o `.pkl`, `predict_proba`, predição em lote, SHAP quando aplicável.

### Parquet, coluna `risco` e o Theo (confirmação)

**Sim:** os dados que o Theo consulta em SQL são **o mesmo conjunto tabular** carregado a partir do **Parquet** (função `load_dados_df` → DataFrame em memória). Esse DataFrame é exposto ao motor **DuckDB** como tabela **`dados`**, e o Theo gera `SELECT` sobre essa tabela.

Fluxo resumido:

1. O Parquet é lido na arranque da app (caminho definido em `get_parquet_path()` — ver secção 4).
2. Ao preparar o *runner* do chat (`app/cached.py` → `make_chat_sql_runner`), o sistema garante a coluna **`risco`**: se já existir no ficheiro (por exemplo após o `etl.py` ter corrido com o modelo presente), usa-se; caso contrário, **`ensure_risco_column`** calcula `risco` com **`predict_risk_probabilities`** sobre esse mesmo DataFrame.
3. O Theo, nas perguntas quantitativas, executa SQL **apenas de leitura** sobre esse DataFrame — portanto pode filtrar, ordenar e agregar por **`risco`**, INDE, Fase, etc., **em cima dos mesmos dados** (mais o dicionário e textos institucionais como contexto da LLM).

Ou seja: **não** há uma segunda base “só para o modelo” e outra “só para o Theo”; o modelo alimenta a coluna de risco e o **Theo interroga a visão tabular unificada** (`dados`) que inclui essa coluna quando disponível.

---

## 6. Assistente Theo (chat analítico)

- O utilizador formula perguntas em linguagem natural.
- O sistema constrói contexto a partir do **dicionário**, opcionalmente **`resumo_anual.txt`**, e do texto do relatório Gamma em **`assets/relatorio_gamma_context.md`** (ou override **`.passos_gamma_context.txt`** na raiz, ignorado pelo Git se configurado).
- Para perguntas que pedem só narrativa institucional (ex.: resumo, relatório, Gamma), o encaminhamento em **`passos_magico/llm/institutional_router.py`** evita SQL desnecessário.
- Para perguntas quantitativas, o fluxo gera **SQL** sobre a tabela **`dados`** no DuckDB — **os mesmos dados do Parquet**, incluindo **`risco`** quando calculada ou já persistida (ver secção 5, último subcapítulo) — e pode produzir **gráficos Plotly**.

Configuração Ollama: **`passos_magico/llm/config.py`** (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_NUM_CTX`, `OLLAMA_REQUEST_TIMEOUT` em segundos por pedido, ou variáveis de ambiente homónimas).

---

## 7. Páginas da aplicação Streamlit

| Página | Função |
|--------|--------|
| **Chat analítico** | Theo + SQL + gráficos |
| **Relatório anual** | Site Gamma embebido por iframe; URL por defeito configurável |
| **Previsão de risco** | Ficha individual (SHAP, Theo) e **matriz de priorização** (filtros, métricas de recorte, exportação CSV); simulação em modo técnico |
| **Dashboards** | KPIs e gráficos (estilo claro); parecer do Theo em `data/dashboard_theo_feedback.txt`, regenerado só quando os dados mudam |
| **Dicionário de dados** | Edição do dicionário e pré-visualização Parquet/CSV |

### Relatório anual (Gamma)

- URL por defeito no código: `https://datathon-passos-magicos-sojo7d1.gamma.site/` (apresentação Gamma do desafio).
- Personalização: variável **`PM_RELATORIO_ANUAL_GAMMA_URL`** ou secret Streamlit **`RELATORIO_ANUAL_GAMMA_URL`**.
- O Theo **não lê a página em tempo real**; usa o ficheiro Markdown em **`assets/relatorio_gamma_context.md`** — deve ser atualizado quando o conteúdo do site mudar, para manter coerência das respostas.

---

## 8. Pré-requisitos

1. **Windows** (o arranque documentado usa `run.bat`; em Linux/macOS adapte os comandos).
2. **Python 3.10+** (`py` ou `python` no PATH).
3. **Ollama** — ver secção **8.1** (instalação + modelo + teste).
4. Ficheiro **`data/relatorio.csv`** presente (o repositório pode incluir um exemplo para desenvolvimento).

### 8.1 Ollama: do download ao teste com o **mesmo modelo** da app

O código usa por defeito o modelo **`llama3`** no endpoint local (`passos_magico/llm/config.py` → `OLLAMA_MODEL`, predefinição `llama3`).

1. **Instalar Ollama** (Windows): descarregar o instalador em [ollama.com/download](https://ollama.com/download), instalar e confirmar que a app **Ollama** fica a correr em segundo plano (ícone na bandeja).
2. **Descarregar o modelo** (PowerShell ou terminal):

   ```bash
   ollama pull llama3
   ```

   Se preferires outro modelo suportado pelo Ollama, define **`OLLAMA_MODEL=nome_do_modelo`** antes de iniciar o Streamlit (tem de existir em `ollama list`).
3. **Verificar o serviço:** no browser, abrir `http://127.0.0.1:11434/` — deve responder (API local).
4. **Teste rápido no terminal:**

   ```bash
   ollama run llama3 "Diz olá em uma frase."
   ```

   Se responder, o motor está OK.
5. **Alinhar com a app:** com o mesmo nome em `OLLAMA_MODEL` (ou `llama3` por defeito), arranca o Streamlit (secção 9). Na página **Chat analítico**, faz uma pergunta simples (ex.: *«Quantos registos existem na base no total?»* — ver `data/theo_20_smoke_perguntas.txt` para uma lista de 20 smoke tests).
6. **Timeouts:** pedidos longos podem estourar o tempo máximo; aumenta **`OLLAMA_REQUEST_TIMEOUT`** (segundos) se o PC for mais lento.

**Testes automáticos com Ollama** (opcional, requer LangChain + Ollama a correr):

```bash
pytest tests/test_theo_e2e_optional.py -m ollama -v
```

---

## 9. Como executar

1. Clonar ou copiar o projeto e abrir a pasta na raiz do repositório.
2. Colocar ou atualizar **`data/relatorio.csv`**.
3. Garantir **`modelo_risco_aluno.pkl`** na raiz (treinar pelo notebook se necessário).
4. Executar **`run.bat`** (duplo clique) **ou**, manualmente (o ficheiro **`.streamlit/config.toml`** fixa o **tema escuro** por defeito no Streamlit):

   ```bat
   py -3 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python scripts\etl.py
   python -m streamlit run app\main.py
   ```

O `run.bat` cria o `.venv`, instala dependências, corre o ETL se faltar `data\dados.parquet` e, se o Parquet já existir e houver **`modelo_risco_aluno.pkl`** na raiz, volta a correr o ETL para atualizar a coluna **`risco`**. Depois inicia o Streamlit com `PYTHONPATH` na raiz do projeto.

Atalho com ícone: na raiz existe **`Passos Mágicos.lnk`** (criado via script PowerShell referenciado no `run.bat`, se aplicável no teu ambiente).

---

## 10. Variáveis de ambiente e secrets (local + Streamlit Cloud)

### Variáveis de ambiente úteis

| Variável | Efeito |
|----------|--------|
| `PASSOS_PARQUET` | Caminho absoluto para o Parquet a usar em vez do defeito |
| `OLLAMA_MODEL` | Nome do modelo no Ollama (ex.: `llama3`) |
| `OLLAMA_BASE_URL` | URL do serviço Ollama (predefinição `http://127.0.0.1:11434`) |
| `OLLAMA_NUM_CTX` | Tamanho do contexto (tokens) |
| `OLLAMA_REQUEST_TIMEOUT` | Tempo máximo (segundos) por chamada ao Ollama (predefinição 120) |
| `PM_RELATORIO_ANUAL_GAMMA_URL` | URL do relatório anual embebido |

### Streamlit Cloud (`Secrets`)

No painel da app → **Settings → Secrets**, podes definir (se aplicável) chaves com o **mesmo nome** das variáveis acima. Exemplo em TOML:

```toml
# Caminho opcional no servidor Cloud (se carregares um Parquet para lá)
# PASSOS_PARQUET = "/mount/.../dados.parquet"

OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
```

**Nota:** no **Streamlit Community Cloud** o processo da app **não** vê o Ollama no teu PC — o Theo em LLM só funciona se tiveres um endpoint Ollama **alcançável a partir da internet** (não documentado por defeito) ou se usares **MVP sem chat LLM** (dashboards + risco + dados). Para vídeo com Theo completo, grava **local** ou numa VM onde Ollama e a app corram juntos.

### Deploy reprodutível e dry-run

1. Fixa dependências: `requirements.txt` já usa **versões fixas** (`==`); após instalar numa máquina limpa, podes gerar um *lock* completo com `pip freeze > requirements-lock.txt` para arquivo.
2. **Dry-run da demo:** usa o **mesmo** `data/dados.parquet` (ou `PASSOS_PARQUET`) que vais mostrar no vídeo; corre `python scripts/etl.py` se partires do CSV; abre todas as páginas críticas (Chat, Risco, Dashboards) uma vez antes de gravar.
3. Garante **`modelo_risco_aluno.pkl`** e **`dicionario.json`** na raiz conforme o teu ambiente de apresentação.

---

## 11. Testes automatizados

Na raiz, com o ambiente ativo:

```bash
pytest tests/
```

Inclui testes de SQL, catálogo de perguntas, risco, simulação e componentes do assistente.

Testes **opcionais** que chamam o Ollama local (marcador `ollama` em `pytest.ini`; fazem `skip` se o serviço não estiver disponível):

```bash
pytest tests/ -m ollama
```

**20 perguntas smoke** (catálogo interno + ficheiro para colar no chat): `data/theo_20_smoke_perguntas.txt`. Teste só de catálogo (sem rede): `pytest tests/test_theo_smoke_catalog.py`.

---

## 12. Privacidade e ética de dados

- Dados de alunos e inferência ML podem processar-se **localmente**.
- O uso de LLM externo em nuvem **não** faz parte do fluxo por defeito; apenas Ollama local.
- Em contexto académico, dados reais devem ser tratados conforme a política da instituição (anonimização, consentimento, retenção).

---

## 13. Estrutura de pastas (resumo)

| Caminho | Conteúdo |
|---------|----------|
| `app/` | Entrada Streamlit, cache |
| `.streamlit/config.toml` | Tema **escuro** por defeito (`base = "dark"`), cores alinhadas à UI |
| `passos_magico/` | Pacote principal (dados, LLM, ML, UI) |
| `data/` | `relatorio.csv`, `dados.parquet`; opcionalmente `dashboard_theo_feedback.txt` (parecer Theo sobre dashboards, cache) |
| `assets/` | Ícones, slogan, contexto Markdown do relatório Gamma |
| `notebooks/Ml/` | Notebook de ML, dados de exemplo, Parquet opcional |
| `scripts/` | `etl.py` (CSV → Parquet, coluna `risco` se o modelo existir) |
| `tests/` | Testes `pytest` |
| Raiz | `modelo_risco_aluno.pkl`, `dicionario.json`, `requirements.txt`, `run.bat` |

---

## 14. Referências e continuidade do trabalho

- Notebook de ML: `notebooks/Ml/ML_Passos_Magicos.ipynb`
- Dependências: `requirements.txt`
- Para evoluções futuras: métricas de modelo no notebook, testes A/B de limiar operacional, pipeline CI, ou documentação de negócio em anexo à entrega académica.

---





#DataScience #MachineLearning #Educação #Streamlit #OpenSource #ImpactoSocial

---

**Post (versão mais técnica)**

Publiquei [INSERIR LINK] um MVP que integra **consulta analítica em PT** (LLM → SQL em DuckDB sobre Parquet), **dashboards**, **modelo de risco escolar** com **SHAP** e leitura de **contexto textual** (relatório Gamma em Markdown). O desenho privilegia **governança**: limiares explícitos na UI, copy orientada a direção pedagógica e opção de correr **sem** dependências de LLM na nuvem.

Ferramentas: Streamlit, Ollama (`llama3` por defeito), LangChain, imbalanced-learn, XGBoost.

Aberto a críticas construtivas e a ideias para evolução (API, RBAC, multi-instância).

#MLOps #ExplainableAI #EdTech #Python

---

*Entrega desenvolvida no âmbito da formação em dados e impacto social — Passos Mágicos / FIAP.*

**Made by Joe Allan Zirn**
