"""
Exportacao de artefato unico para inferencia por aluno (pipeline + metadados + template).
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Any, Mapping, Optional

import pandas as pd


def exportar_artefato_inferencia_aluno(
    *,
    pipeline_melhor: Any,
    modelo_nome: str,
    all_features: list[str],
    feat_num: list[str],
    feat_cat: list[str],
    X_referencia: pd.DataFrame,
    caminho_saida: str = "artefatos/inferencia_modelagem.pkl",
    df_resultados: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    risk_df: Optional[pd.DataFrame] = None,
    metadata_extras: Optional[Mapping[str, Any]] = None,
    template_csv_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Grava um pickle com pipeline treinado e tudo que precisa para montar linhas de aluno
    e chamar predict / predict_proba.

    Parametros
    ----------
    X_referencia : DataFrame com pelo menos as colunas em `all_features` (ex.: X_train).
        Usado para gerar `template_inferencia`: DataFrame vazio com mesma ordem de colunas
        e dtypes do treino — base para preencher um aluno e empilhar antes do pipeline.

    template_csv_path : se informado, grava CSV so com cabecalho (mesma ordem de features),
        util para preenchimento manual ou ETL externo.
    """
    faltando = [c for c in all_features if c not in X_referencia.columns]
    if faltando:
        raise ValueError(
            f"X_referencia nao contem colunas esperadas em all_features: {faltando[:10]}"
            + (" ..." if len(faltando) > 10 else "")
        )

    template_inferencia = X_referencia[all_features].iloc[:0].copy()

    metadata: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_definicao": "1=sucesso, 0=insucesso",
        "metrica_prioritaria": "F2-Score Risco (classe 0)",
        "instrucao_inferencia": (
            "Monte um DataFrame com uma linha por aluno, colunas em `all_features`, "
            "mesmos dtypes que `template_inferencia`; use pipeline.predict_proba(X)."
        ),
    }
    if metadata_extras:
        metadata.update(dict(metadata_extras))

    artefato: dict[str, Any] = {
        "modelo_nome": modelo_nome,
        "pipeline_melhor": pipeline_melhor,
        "all_features": list(all_features),
        "feat_num": list(feat_num),
        "feat_cat": list(feat_cat),
        "template_inferencia": template_inferencia,
        "df_resultados": df_resultados.copy() if df_resultados is not None else None,
        "X_test": X_test.copy() if X_test is not None else None,
        "y_test": y_test.copy() if y_test is not None else None,
        "risk_df": risk_df.copy() if risk_df is not None else None,
        "metadata": metadata,
    }

    out_dir = os.path.dirname(os.path.abspath(caminho_saida))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(caminho_saida, "wb") as f:
        pickle.dump(artefato, f, protocol=pickle.HIGHEST_PROTOCOL)

    if template_csv_path:
        tdir = os.path.dirname(os.path.abspath(template_csv_path))
        if tdir:
            os.makedirs(tdir, exist_ok=True)
        template_inferencia.to_csv(template_csv_path, index=False)

    return artefato


def carregar_artefato_inferencia(caminho: str) -> dict[str, Any]:
    """Carrega o pickle gerado por `exportar_artefato_inferencia_aluno`."""
    with open(caminho, "rb") as f:
        return pickle.load(f)


def montar_X_alunos(
    artefato: Mapping[str, Any], linhas: list[Mapping[str, Any]] | pd.DataFrame
) -> pd.DataFrame:
    """
    Constroi DataFrame alinhado ao treino a partir de dicts ou de um DataFrame.
    Colunas faltantes viram NA; colunas extras sao ignoradas.
    """
    cols: list[str] = list(artefato["all_features"])

    if isinstance(linhas, pd.DataFrame):
        X = linhas.reindex(columns=cols)
    else:
        X = pd.DataFrame(linhas)
        X = X.reindex(columns=cols)
    return X


def probabilidade_sucesso(artefato: Mapping[str, Any], X: pd.DataFrame) -> pd.Series:
    """Retorna P(classe 1 = sucesso) para cada linha de X."""
    pipe = artefato["pipeline_melhor"]
    proba = pipe.predict_proba(X)
    # classe positiva = indice 1 em classificacao binaria sklearn
    return pd.Series(proba[:, 1], index=X.index, name="p_sucesso")
