# analise-preditiva-desempenho-academico
Projeto de Machine Learning supervisionado voltado à predição de risco de reprovação acadêmica, utilizando dados educacionais anonimizados. O objetivo é identificar, de forma antecipada, estudantes com maior probabilidade de reprovação, permitindo a geração de insights e o apoio à tomada de decisão institucional.

## Dados

Os dados utilizados neste projeto são de caráter institucional e não estão disponíveis publicamente.

Para execução do notebook, é necessário inserir os arquivos `.parquet` na pasta local.

## Pipeline de notebooks (recomendado)

Execute na ordem:

1. **`01_limpeza_transformacao.ipynb`** — carrega os Parquets, limpa, integra tabelas, define o alvo, gera features (sem one-hot) e exporta `dados/df_modelo_tratado.csv`.
2. **`02_analise_exploratoria.ipynb`** — lê o CSV e produz gráficos/tabelas de EDA.
3. **`03_modelagem_previsao.ipynb`** — lê o CSV, aplica `make_dummie`, treina e avalia os modelos.

Para regenerar os três arquivos a partir do notebook monolítico: `python scripts/split_notebooks.py`

O arquivo único `ebtt_analise_preditiva_v2 (1).ipynb` permanece como referência consolidada.
