"""
Listagem de matriculas em risco: id do discente + disciplina, com filtros institucionais.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard_io import BASE_CSV, BASE_PARQUET, load_base

st.set_page_config(page_title="Alunos em risco", page_icon="⚠️", layout="wide")

st.title("Alunos em risco")
st.caption(
    "Matriculas com maior probabilidade de insucesso prevista pelo modelo. "
    "Use os filtros para recortar por centro, departamento e curso."
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

# Normaliza textos para filtros (copia para nao alterar cache)
for col in ("sigla_centro", "sigla_departamento", "curso_nome"):
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": "Nao informado", "None": "Nao informado"})
        )

st.sidebar.header("Filtros desta pagina")

centros = sorted(df["sigla_centro"].dropna().unique()) if "sigla_centro" in df.columns else []
deps = sorted(df["sigla_departamento"].dropna().unique()) if "sigla_departamento" in df.columns else []
cursos = sorted(df["curso_nome"].dropna().unique()) if "curso_nome" in df.columns else []

sel_centro = st.sidebar.multiselect("Sigla do centro", centros, default=[])
sel_dep = st.sidebar.multiselect("Departamento (sigla)", deps, default=[])
sel_curso = st.sidebar.multiselect("Curso", cursos, default=[])

_corte_opcoes = {
    "Somente faixa Alto": "alto",
    "Faixa Alto e Medio": "alto_medio",
    "Todas as faixas": "todas",
}
_corte_label = st.sidebar.selectbox("Recorte de risco", list(_corte_opcoes.keys()))
corte = _corte_opcoes[_corte_label]

q_min = st.sidebar.number_input("Minimo p_risco_insucesso (opcional)", 0.0, 1.0, 0.0, 0.05)
q_busca = st.sidebar.text_input("Buscar nome da disciplina (contem)", "")

filt = df.copy()
if sel_centro:
    filt = filt[filt["sigla_centro"].isin(sel_centro)]
if sel_dep:
    filt = filt[filt["sigla_departamento"].isin(sel_dep)]
if sel_curso:
    filt = filt[filt["curso_nome"].isin(sel_curso)]

if corte == "alto":
    filt = filt[filt["faixa_risco"] == "Alto"]
elif corte == "alto_medio":
    filt = filt[filt["faixa_risco"].isin(["Alto", "Medio"])]

if q_min > 0:
    filt = filt[filt["p_risco_insucesso"].astype(float) >= q_min]

if q_busca.strip():
    mask = filt["nome_disciplina"].astype(str).str.contains(q_busca.strip(), case=False, na=False)
    filt = filt[mask]

filt = filt.sort_values("p_risco_insucesso", ascending=False)

cols_show = [
    "id_discente",
    "nome_disciplina",
    "p_risco_insucesso",
    "faixa_risco",
    "periodo_letivo",
    "sigla_centro",
    "sigla_departamento",
    "curso_nome",
]
cols_show = [c for c in cols_show if c in filt.columns]

out = filt[cols_show].copy()
out["id_discente"] = out["id_discente"].astype(str)
if "p_risco_insucesso" in out.columns:
    out["p_risco_insucesso"] = out["p_risco_insucesso"].astype(float).round(4)

m1, m2 = st.columns(2)
m1.metric("Matriculas listadas", f"{len(out):,}".replace(",", "."))
if len(out) and "p_risco_insucesso" in out.columns:
    m2.metric("Risco medio (lista)", f"{out['p_risco_insucesso'].mean() * 100:.1f}%")
else:
    m2.metric("Risco medio (lista)", "—")

st.dataframe(out, use_container_width=True, hide_index=True, height=480)

st.download_button(
    "Baixar CSV",
    data=out.to_csv(index=False).encode("utf-8-sig"),
    file_name="alunos_em_risco_filtrado.csv",
    mime="text/csv",
)
