# Manual de instalação e uso

## Requisitos de ambiente

1. **Windows 10 ou superior** (o projeto foi testado com script em lote `run.bat`).
2. **Python 3.10 ou superior**, com o launcher `py` ou o executável `python` disponível no PATH.
3. **Ollama** instalado a partir de [https://ollama.com](https://ollama.com) e em execução na bandeja do sistema.
4. Pelo menos um **modelo** baixado localmente, por exemplo:

   ```text
   ollama pull llama3
   ```

5. Conferir em `passos_magico/llm/config.py` se o nome do modelo e a URL base (`OLLAMA_BASE_URL`) correspondem à sua instalação.

## Instalação rápida (recomendada)

1. Copiar o projeto para uma pasta local (evitar caminhos com permissões restritas se possível).
2. Garantir que `data/relatorio.csv` exista ou deixar o ETL gerar o exemplo sintético na primeira rodada.
3. Dar **duplo clique** em `run.bat` na raiz.

O script realiza, em sequência:

- Criação do ambiente virtual `.venv` se necessário.
- Atualização do `pip` e instalação de `requirements.txt`.
- Execução do ETL se `data/dados.parquet` não existir.
- Treinamento com `scripts/train_model.py` se `models/modelo.joblib` não existir.
- Nova passagem do ETL se o modelo tiver sido criado (para incluir a coluna `risco`).
- Inicialização do Streamlit em `app/main.py`.

## Instalação manual (alternativa)

Na raiz do projeto:

```text
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\etl.py
python scripts\train_model.py
python scripts\etl.py
set PYTHONPATH=%CD%
python -m streamlit run app\main.py --server.headless true
```

Ajuste `PYTHONPATH` no PowerShell com `$env:PYTHONPATH = (Get-Location).Path` se usar esse shell.

## Verificação pós-instalação

| Verificação | Como testar |
|-------------|----------------|
| Ollama ativo | Navegador ou `curl` em `http://localhost:11434/api/tags` |
| Parquet gerado | Arquivo `data/dados.parquet` presente |
| Modelo gerado | Arquivo `models/modelo.joblib` presente |
| Interface | Navegador abre endereço local exibido pelo Streamlit (porta padrão 8501) |

## Uso das três áreas

1. **Chat analítico**: faça perguntas sobre médias, filtros por turma/fase/ano, etc. Verifique se o SQL retornado faz sentido antes de tomar decisões.
2. **Previsão de risco**: selecione aluno ou faça triagem; use SHAP para entender o peso de cada indicador.
3. **Dicionário de dados**: ajuste descrições e salve para refinar o comportamento do assistente nas próximas perguntas.

## Problemas comuns

- **Ollama não encontrado**: iniciar o aplicativo Ollama; confirmar firewall local.
- **Modelo não baixado**: executar `ollama pull` com o mesmo nome configurado em `config.py`.
- **Erro de importação**: garantir que o comando foi executado a partir da **raiz** do projeto ou que `PYTHONPATH` aponta para ela.
- **Gráfico não exporta PNG**: dependência Kaleido no `requirements.txt`; reinstalar o ambiente virtual se necessário.
