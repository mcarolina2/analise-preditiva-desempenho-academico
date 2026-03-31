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

## Dashboard analítico

Foi adicionado um dashboard em Streamlit para análise de aprovações, reprovações e risco:

- Arquivo: `dashboard_analitico.py`
- Fonte de dados: `dados/saida_analise_risco_modelo.csv`

Principais visões do painel:

- Disciplinas críticas (retenção, risco médio, alto risco).
- Áreas de conhecimento com maior falha no primeiro período e nos períodos seguintes.
- Disciplinas com maior impacto potencial em cancelamento (proxy analítico).
- Perfil do aluno (geral, aprovados e reprovados).
- Matriz de confusão e distribuição de risco do modelo.

Como executar:

1. Instale as dependências:
   - `pip install -r requirements.txt`
2. Execute o dashboard:
   - `streamlit run dashboard_analitico.py`
