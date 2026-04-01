"""
Enriquece fato de risco + base modelagem para o dashboard Streamlit.

Gera `dados/base_dashboard_risco.csv` (e Parquet se `fastparquet` estiver instalado) com:
- chaves e scores do modelo;
- periodo letivo (ano.periodo), contagem de componentes no semestre do aluno;
- semestre relativo no curso (ordem temporal por aluno/curso);
- faixas de risco (Alto / Medio / Baixo);
- taxa de reprovacao historica antes da matricula (mesma area).

Execute antes do dashboard:
    python calcular_features_dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Limites de faixa (probabilidade de insucesso). Ajuste institucional.
RISCO_ALTO_MIN = 0.50
RISCO_MEDIO_MIN = 0.25

FATO_PATH = Path("dados/fato_risco_matricula.csv")
MODELO_PATH = Path("dados/df_modelo_tratado.csv")
OUT_CSV = Path("dados/base_dashboard_risco.csv")
OUT_PARQUET = Path("dados/base_dashboard_risco.parquet")


def _periodo_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").fillna(0).astype(int)


def classificar_faixa_risco(p: pd.Series) -> pd.Series:
    """Classifica p_risco_insucesso em Alto / Medio / Baixo."""
    out = pd.Series(np.where(p >= RISCO_ALTO_MIN, "Alto", ""), index=p.index, dtype=object)
    out = np.where((p >= RISCO_MEDIO_MIN) & (p < RISCO_ALTO_MIN), "Medio", out)
    out = np.where(p < RISCO_MEDIO_MIN, "Baixo", out)
    return pd.Series(out, index=p.index)


def emoji_faixa(serie: pd.Series) -> pd.Series:
    m = {"Alto": "🔴", "Medio": "🟡", "Baixo": "🟢"}
    return serie.map(m).fillna("")


def main() -> None:
    if not FATO_PATH.exists():
        raise FileNotFoundError(f"Gere antes o fato: {FATO_PATH}")
    if not MODELO_PATH.exists():
        raise FileNotFoundError(f"Base modelagem ausente: {MODELO_PATH}")

    base = pd.read_csv(MODELO_PATH, low_memory=False)
    base["id_linha_base"] = np.arange(len(base), dtype=np.int64)

    scores = pd.read_csv(FATO_PATH, low_memory=False)
    cols_score = [
        "id_linha_base",
        "sk_matricula_composta",
        "p_sucesso",
        "p_risco_insucesso",
        "modelo_nome",
        "data_geracao_score",
        "target_observado",
    ]
    missing = [c for c in cols_score if c not in scores.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em fato: {missing}")

    df = base.merge(scores[cols_score], on="id_linha_base", how="inner", validate="one_to_one")

    df["nome_disciplina"] = df["nome_componete_curricular"].astype(str).str.strip()
    df["area_conhecimento"] = df["area_conhecimento"].astype(str).str.strip().replace({"nan": "Nao informado"})
    df["sigla_centro"] = df["sigla_centro"].astype(str).str.strip()

    df["periodo_n"] = _periodo_num(df["periodo"])
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["periodo_letivo"] = df["ano"].astype("Int64").astype(str) + "." + df["periodo_n"].astype(str)
    df["periodo_letivo_ord"] = df["ano"] * 10 + df["periodo_n"]

    # Quantidade de componentes (disciplinas) do aluno naquele semestre = ano.periodo
    df["n_componentes_semestre"] = df.groupby(["id_discente", "periodo_letivo"], sort=False)[
        "id_disciplina"
    ].transform("count")

    # Semestre relativo no curso: 1 = primeira matricula daquele aluno naquele curso, por tempo
    ord_cols = ["id_discente", "id_curso", "ano", "periodo_n", "id_disciplina"]
    df = df.sort_values(ord_cols, kind="mergesort")
    df["semestre_rel_curso"] = df.groupby(["id_discente", "id_curso"], sort=False).cumcount() + 1

    df["faixa_risco"] = classificar_faixa_risco(df["p_risco_insucesso"].astype(float))
    df["faixa_risco_emoji"] = emoji_faixa(df["faixa_risco"])

    # Taxa de reprovacao no historico (mesma area) antes desta linha
    na = pd.to_numeric(df["n_aprov_antes_mesma_area"], errors="coerce").fillna(0)
    nr = pd.to_numeric(df["n_reprov_antes_mesma_area"], errors="coerce").fillna(0)
    nt = pd.to_numeric(df["n_tranc_desist_antes_mesma_area"], errors="coerce").fillna(0)
    denom = na + nr + nt
    df["taxa_reprov_historica_antes"] = np.where(denom > 0, nr / denom, np.nan)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Gravado: {OUT_CSV} ({len(df):,} linhas)")
    try:
        df.to_parquet(OUT_PARQUET, index=False, engine="fastparquet")
        print(f"Gravado: {OUT_PARQUET}")
    except (ImportError, ValueError, OSError):
        print("(Parquet opcional nao gerado — use CSV ou pip install fastparquet)")
    print(f"Faixas: Alto>={RISCO_ALTO_MIN}, Medio>={RISCO_MEDIO_MIN}, Baixo<{RISCO_MEDIO_MIN}")


if __name__ == "__main__":
    main()
