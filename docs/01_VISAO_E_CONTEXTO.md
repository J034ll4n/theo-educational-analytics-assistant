# Visão e contexto

## Problema

Instituições educacionais que acompanham alunos por meio de relatórios com indicadores (por exemplo INDE, IDA, IAN, IEG, IPV) precisam cruzar essas informações de forma ágil para responder perguntas como médias por turma, evolução por ano ou comparação entre fases. Ao mesmo tempo, é útil estimar **risco pedagógico** com base nesses mesmos sinais, de modo a priorizar acompanhamento — sempre com transparência sobre como o modelo chegou a cada conclusão.

## Objetivos do projeto

1. **Centralizar** os dados tabulares em um formato eficiente para consulta (Parquet) e exploração.
2. **Oferecer um assistente conversacional** (“Theo”) que traduza perguntas em linguagem natural em SQL, execute sobre os dados locais e apresente resultados tabulares e gráficos.
3. **Treinar um classificador** que estime probabilidade de “alto risco” a partir dos indicadores, com explicação por atributo (SHAP) e ferramentas de simulação e triagem em lote na interface.
4. **Manter privacidade**: processamento local (navegador + processo Python + Ollama na máquina), sem envio obrigatório de dados sensíveis a serviços externos.

## Escopo

- **Incluído**: ingestão a partir de CSV; normalização de tipos; painel Streamlit com três áreas (chat analítico, previsão de risco, dicionário de dados editável); geração de SQL apenas `SELECT`; visualizações Plotly; modelo Random Forest persistido em disco; documentação de uso.
- **Fora do escopo nesta versão**: autenticação multiusuário, API REST pública, deploy em nuvem, retreino automático agendado ou integração com sistemas legados da escola.

## Público-alvo

Equipe pedagógica ou técnica que já trabalha com planilhas ou relatórios exportados e deseja reduzir o tempo entre a pergunta e a visualização, além de obter uma **estimativa objetiva de risco** para apoiar decisões — sempre sujeita à validação humana.

## Alinhamento com a Postech (Fase 5)

A Fase 5 costuma consolidar um produto com **dados reais ou representativos**, **inteligência aplicada** (aqui: LLM local + ML supervisionado) e **entrega executável**. Este projeto demonstra pipeline completo (dados → modelo → interface), critérios explícitos de rótulo para o ML e camadas de explicabilidade e governança mínima (SQL restrito, dicionário editável).
