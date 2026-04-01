"""
Visao geral: volume de matriculas por periodo, taxa de reprovacao e perfil do discente.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard_io import BASE_CSV, BASE_PARQUET, load_base

st.set_page_config(page_title="Visao geral", page_icon="📈", layout="wide")

sns.set_theme(style="whitegrid", context="notebook")

st.title("Visao geral")
st.caption(
    "Panorama institucional na base de modelagem: matriculas por periodo letivo, taxa de reprovacao "
    "(por matricula em disciplina) e perfil demografico dos **discentes unicos**."
)

if not BASE_CSV.exists() and not BASE_PARQUET.exists():
    st.error(
        f"Base nao encontrada. Execute: `python calcular_features_dashboard.py` "
        f"(esperado `{BASE_CSV}`)."
    )
    st.stop()

try:
    df = load_base().copy()
except Exception as e:
    st.error(f"Erro ao carregar base: {e}")
    st.stop()


def _col_alvo(frame: pd.DataFrame) -> pd.Series:
    if "target" in frame.columns:
        return pd.to_numeric(frame["target"], errors="coerce")
    if "target_observado" in frame.columns:
        return pd.to_numeric(frame["target_observado"], errors="coerce")
    raise ValueError("Coluna de resultado (target) ausente na base.")


# --- Filtros opcionais ---
st.sidebar.header("Filtros (opcional)")
for col in ("sigla_centro", "curso_nome"):
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": "Nao informado", "None": "Nao informado"})
        )

centros = sorted(df["sigla_centro"].dropna().unique()) if "sigla_centro" in df.columns else []
cursos = sorted(df["curso_nome"].dropna().unique()) if "curso_nome" in df.columns else []
sel_c = st.sidebar.multiselect("Centro", centros, default=[])
sel_cur = st.sidebar.multiselect("Curso", cursos, default=[])

d = df.copy()
if sel_c:
    d = d[d["sigla_centro"].isin(sel_c)]
if sel_cur:
    d = d[d["curso_nome"].isin(sel_cur)]

d["_alvo"] = _col_alvo(d)
d_valid = d.dropna(subset=["_alvo"])

# --- KPIs ---
st.subheader("Indicadores rapidos")
n_mat = len(d_valid)
n_disc = d_valid["id_discente"].nunique()
taxa_rep = (1.0 - d_valid["_alvo"].mean()) * 100 if n_mat else 0.0
c1, c2, c3 = st.columns(3)
c1.metric("Matriculas (linhas)", f"{n_mat:,}".replace(",", "."))
c2.metric("Discentes unicos", f"{n_disc:,}".replace(",", "."))
c3.metric("Taxa de reprovacao (geral)", f"{taxa_rep:.1f}%")
st.caption(
    "Taxa de reprovacao = proporcao de matriculas com insucesso (`target` = 0) entre as linhas com alvo informado."
)

st.divider()

# --- 1) Matriculas por periodo x taxa de reprovacao ---
st.subheader("Matriculas por periodo letivo e taxa de reprovacao")
st.caption("Cada ponto no eixo X e um `ano.periodo`. Barras = volume; linha = taxa de reprovacao (%).")

if "periodo_letivo" not in d_valid.columns:
    st.warning("Coluna `periodo_letivo` ausente.")
else:
    po = (
        d_valid.drop_duplicates(subset=["periodo_letivo"])[["periodo_letivo", "periodo_letivo_ord"]]
        if "periodo_letivo_ord" in d_valid.columns
        else None
    )
    g = (
        d_valid.groupby("periodo_letivo", as_index=False)
        .agg(n_matriculas=("id_discente", "count"), taxa_reprov=("_alvo", lambda s: (1.0 - s.mean()) * 100))
    )
    if po is not None:
        g = g.merge(po, on="periodo_letivo", how="left").sort_values("periodo_letivo_ord")
    else:
        g = g.sort_values("periodo_letivo")

    if g.empty:
        st.info("Sem dados agregados para o grafico de periodos.")
    else:
        fig, ax1 = plt.subplots(figsize=(12, 4.5))
        x = np.arange(len(g))
        ax1.bar(x, g["n_matriculas"], color="steelblue", alpha=0.75, label="Matriculas")
        ax1.set_ylabel("N. de matriculas")
        ax1.set_xticks(x)
        ax1.set_xticklabels(g["periodo_letivo"].astype(str), rotation=45, ha="right", fontsize=8)
        ax1.set_xlabel("Periodo letivo")

        ax2 = ax1.twinx()
        ax2.plot(x, g["taxa_reprov"], color="darkred", marker="o", linewidth=2, markersize=5, label="Taxa reprov. (%)")
        ax2.set_ylabel("Taxa de reprovacao (%)")
        ymax = float(g["taxa_reprov"].max()) if len(g) else 5.0
        ax2.set_ylim(0, max(ymax * 1.15, 5.0))

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.divider()

# --- Perfil: discente unico (primeira ocorrencia por id_discente) ---
st.subheader("Perfil do aluno (discentes unicos)")
st.caption(
    "Proporcoes calculadas sobre **um registro por `id_discente`** (primeira linha apos filtros), "
    "para nao multiplicar o mesmo aluno por varias matriculas."
)

d_demo = d_valid.sort_values(["id_discente", "periodo_letivo_ord"] if "periodo_letivo_ord" in d_valid.columns else ["id_discente"]).drop_duplicates(
    subset=["id_discente"], keep="first"
)
n_demo = len(d_demo)
if n_demo == 0:
    st.info("Sem discentes apos filtros.")
    st.stop()

st.metric("Base do perfil", f"{n_demo:,} discentes".replace(",", "."))


def prop_table(serie: pd.Series, nome: str) -> pd.DataFrame:
    s = serie.astype(str).str.strip().replace({"nan": "Nao informado", "None": "Nao informado"})
    t = s.value_counts(dropna=False)
    out = (
        pd.DataFrame({nome: t.index, "quantidade": t.values, "percentual": (t.values / n_demo * 100).round(2)})
        .reset_index(drop=True)
    )
    return out


# Sexo
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Sexo**")
    if "sexo" in d_demo.columns:
        t_sexo = prop_table(d_demo["sexo"], "sexo")
        st.dataframe(t_sexo, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(t_sexo["quantidade"], labels=t_sexo["sexo"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Distribuicao por sexo")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Coluna `sexo` ausente.")

# Faixa etaria (a partir de idade)
with c2:
    st.markdown("**Faixa etaria**")
    if "idade" in d_demo.columns:
        idade = pd.to_numeric(d_demo["idade"], errors="coerce")
        bins = [0, 17, 21, 25, 30, 35, 40, 50, 120]
        labels = ["<=17", "18-21", "22-25", "26-30", "31-35", "36-40", "41-50", ">50"]
        faixa = pd.cut(idade, bins=bins, labels=labels, right=True)
        t_id = prop_table(faixa.astype(str), "faixa_etaria")
        st.dataframe(t_id, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=t_id, x="faixa_etaria", y="percentual", color="teal", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("% dos discentes")
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Coluna `idade` ausente.")

# Membros da familia, renda, forma de ingresso
st.markdown("**Demografia e contexto**")
r1, r2, r3 = st.columns(3)

with r1:
    st.markdown("*Membros na familia*")
    if "faixa_membros_familia" in d_demo.columns:
        st.dataframe(prop_table(d_demo["faixa_membros_familia"], "faixa_membros_familia"), use_container_width=True, hide_index=True)
    else:
        st.caption("Coluna ausente.")

with r2:
    st.markdown("*Faixa de renda familiar*")
    if "faixa_renda_familiar" in d_demo.columns:
        st.dataframe(prop_table(d_demo["faixa_renda_familiar"], "faixa_renda_familiar"), use_container_width=True, hide_index=True)
    else:
        st.caption("Coluna ausente.")

with r3:
    st.markdown("*Forma de ingresso*")
    if "forma_ingresso" in d_demo.columns:
        st.dataframe(prop_table(d_demo["forma_ingresso"], "forma_ingresso"), use_container_width=True, hide_index=True)
    else:
        st.caption("Coluna ausente.")

# Raca — largura total
if "raca_declarada" in d_demo.columns:
    st.markdown("**Raca declarada**")
    t_r = prop_table(d_demo["raca_declarada"], "raca_declarada")
    c_a, c_b = st.columns([1, 1])
    with c_a:
        st.dataframe(t_r.head(25), use_container_width=True, hide_index=True)
    with c_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        top = t_r.head(12)
        sns.barplot(data=top, y="raca_declarada", x="percentual", color="coral", ax=ax)
        ax.set_xlabel("% dos discentes")
        st.pyplot(fig)
        plt.close(fig)
