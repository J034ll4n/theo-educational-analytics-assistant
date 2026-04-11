# Modelo de machine learning (risco escolar)

## Objetivo

Estimar a **probabilidade de alto risco pedagógico** por aluno com base em variáveis já presentes no relatório, permitindo triagem, simulação “what-if” e explicação por contribuição de atributos.

## Definição do rótulo (supervisão)

O rótulo binário não vem de um campo do CSV: ele é **derivado por regra explícita** no código (`passos_magico/ml/features.py`, função `build_xy`):

- Classe positiva (1) se **(IDA < 6,5 e IEG < 6,0)** **ou** **INDE < 5,0**.
- Caso contrário, classe negativa (0).

Essa regra funciona como **proxy pedagógico** para treinar o classificador em cima de um snapshot estático. Em uso real, a equipe pode revisar o limiar e a definição com especialistas — o importante para a entrega acadêmica é que o critério esteja documentado e reproduzível.

## Features

Após enriquecimento (`augment_dataframe`):

| Feature | Origem |
|---------|--------|
| `Fase`, `Ano` | Numéricos diretos |
| `INDE`, `IDA`, `IAN`, `IEG`, `IPV` | Indicadores |
| `Turma_ord` | Mapeamento ordinal A→1, B→2, … |
| `Pedra_ord` | Mapeamento ordinal das pedras (Quartzo, Ágata, etc.) |

A ordem fixa `FEATURE_ORDER` garante alinhamento entre treino, inferência e simulador na interface.

## Algoritmo e treino (`scripts/train_model.py`)

- **Algoritmo**: `RandomForestClassifier` (floresta aleatória).
- **Partição**: 75% treino / 25% teste, estratificada, `random_state=42`.
- Hiperparâmetros principais: 160 árvores, profundidade máxima 14, `class_weight='balanced'` para mitigar desbalanceamento.
- **Saída**: `models/modelo.joblib` contendo o classificador e metadados de colunas.

O script imprime `classification_report` no console para inspeção offline (precisão, recall, F1 por classe).

## Inferência e SHAP

- **Probabilidade**: `predict_proba` retorna \(P(\text{alto risco})\) usada na coluna `risco` do Parquet e nos painéis.
- **SHAP**: valores SHAP explicam quanto cada feature empurrou a predição para cima ou para baixo em um caso individual, auxiliando a leitura humana.

## Limitações conscientes

- O rótulo é uma **regra fixa**, não um diagnóstico clínico ou oficial da instituição.
- **Random Forest** captura não linearidades, mas pode refletir vieses presentes nos dados de entrada.
- Generalização para outros anos ou outras unidades exige validação e possivelmente retreino.

## Relação com o ETL

Depois do primeiro treino, convém rodar novamente o ETL para materializar a coluna `risco` no Parquet consumido pelo chat e pelos gráficos SQL. O `run.bat` automatiza essa sequência quando o modelo passa a existir.
