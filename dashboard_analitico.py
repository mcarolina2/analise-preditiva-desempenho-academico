"""
Dashboard Streamlit — pagina principal: previsao de risco por disciplina e periodo.

Outras paginas: `pages/1_Visao_geral.py`, `pages/2_Alunos_em_risco.py`.

Pre-requisito:
    python calcular_features_dashboard.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard_io import BASE_CSV, BASE_PARQUET, load_base

# Estilo graficos
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 12


def sidebar_filtros(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    st.sidebar.header("Filtros")
    _po = df.drop_duplicates(subset=["periodo_letivo"])[["periodo_letivo", "periodo_letivo_ord"]].sort_values("periodo_letivo_ord")
    periodos = _po["periodo_letivo"].dropna().tolist()
    areas = sorted(df["area_conhecimento"].dropna().unique())
    sel_p = st.sidebar.multiselect("Periodo letivo (ano.periodo)", periodos, default=[])
    sel_a = st.sidebar.multiselect("Area de conhecimento", areas, default=[])
    min_n = st.sidebar.number_input("Min. matriculas por disciplina (rankings)", min_value=5, max_value=500, value=30)

    out = df.copy()
    if sel_p:
        out = out[out["periodo_letivo"].isin(sel_p)]
    if sel_a:
        out = out[out["area_conhecimento"].isin(sel_a)]
    return out, min_n


def secao_ranking(d: pd.DataFrame, min_n: int) -> None:
    st.subheader("1. Ranking de risco por disciplina")
    st.caption("Score medio de `p_risco_insucesso` por disciplina, com faixas 🔴 Alto / 🟡 Medio / 🟢 Baixo.")
    g = (
        d.groupby("nome_disciplina", as_index=False)
        .agg(
            n=("id_discente", "count"),
            risco_medio=("p_risco_insucesso", "mean"),
            pct_alto=("faixa_risco", lambda s: (s == "Alto").mean() * 100),
        )
        .query("n >= @min_n")
        .sort_values("risco_medio", ascending=False)
    )
    g["faixa_media"] = np.where(
        g["risco_medio"] >= 0.50, "🔴 Alto", np.where(g["risco_medio"] >= 0.25, "🟡 Medio", "🟢 Baixo")
    )
    top_n = st.slider("Quantidade de disciplinas no ranking", 5, 50, 20, 5)
    show = g.head(top_n).copy()
    show["risco_medio_pct"] = (show["risco_medio"] * 100).round(2)
    show["pct_alto"] = show["pct_alto"].round(1)
    st.dataframe(
        show[["nome_disciplina", "n", "risco_medio_pct", "pct_alto", "faixa_media"]].rename(
            columns={
                "nome_disciplina": "Disciplina",
                "n": "Matriculas",
                "risco_medio_pct": "Risco medio (%)",
                "pct_alto": "% alunos Alto",
                "faixa_media": "Faixa (media)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.22)))
    sns.barplot(data=show, y="nome_disciplina", x="risco_medio", color="steelblue", ax=ax)
    ax.set_xlabel("Risco medio (prob. insucesso)")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    st.pyplot(fig)
    plt.close(fig)


def secao_heatmap(d: pd.DataFrame, min_n: int) -> None:
    st.subheader("2. Mapa de calor — periodo letivo x disciplina")
    st.caption("Intensidade = risco medio. Periodo no eixo X = `ano.periodo` (calendario academico).")
    n_disc = st.slider("Max. disciplinas no mapa (mais frequentes)", 8, 45, 25)
    top_disc = d.groupby("nome_disciplina").size().sort_values(ascending=False).head(n_disc).index
    sub = d[d["nome_disciplina"].isin(top_disc)]
    if sub.empty:
        st.info("Sem dados apos filtros.")
        return
    pivot = sub.pivot_table(
        index="nome_disciplina",
        columns="periodo_letivo",
        values="p_risco_insucesso",
        aggfunc="mean",
    )
    # Ordena colunas por periodo_letivo_ord
    ord_map = d.drop_duplicates("periodo_letivo").set_index("periodo_letivo")["periodo_letivo_ord"].sort_values()
    cols_ord = [c for c in ord_map.index if c in pivot.columns]
    pivot = pivot[cols_ord]
    pivot = pivot.reindex(top_disc)
    fig_h = max(6, n_disc * 0.25)
    fig, ax = plt.subplots(figsize=(min(20, 0.35 * len(cols_ord) + 4), fig_h))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Risco medio"})
    ax.set_xlabel("Periodo letivo (ano.periodo)")
    ax.set_ylabel("Disciplina")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0, fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


def secao_distribuicao(d: pd.DataFrame, min_n: int) -> None:
    st.subheader("3. Distribuicao de risco dos alunos por disciplina")
    st.caption("Boxplot do score individual; tabela com contagem por faixa.")
    candidatas = (
        d.groupby("nome_disciplina")
        .agg(n=("id_discente", "count"))
        .query("n >= @min_n")
        .sort_values("n", ascending=False)
        .head(40)
        .index.tolist()
    )
    escolhidas = st.multiselect("Disciplinas", candidatas, default=candidatas[:6])
    if not escolhidas:
        st.info("Selecione ao menos uma disciplina.")
        return
    sub = d[d["nome_disciplina"].isin(escolhidas)]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=sub, x="nome_disciplina", y="p_risco_insucesso", ax=ax, color="lightblue")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("p_risco_insucesso")
    ax.set_xlabel("")
    ax.axhline(0.5, color="red", ls="--", alpha=0.6, label="limiar Alto")
    ax.axhline(0.25, color="orange", ls="--", alpha=0.6, label="limiar Medio")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    cnt = (
        sub.groupby(["nome_disciplina", "faixa_risco"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Baixo", "Medio", "Alto"], fill_value=0)
    )
    st.dataframe(cnt, use_container_width=True)


def secao_concentracao_alto(d: pd.DataFrame, min_n: int) -> None:
    st.subheader("4. Disciplinas com maior concentracao de alunos em risco Alto")
    st.caption("Barras empilhadas: percentual de matriculas em cada faixa (por disciplina).")
    n_show = st.slider("Numero de disciplinas", 5, 30, 15)
    g = (
        d.groupby("nome_disciplina", observed=True)
        .agg(n=("id_discente", "count"))
        .query("n >= @min_n")
        .index
    )
    sub = d[d["nome_disciplina"].isin(g)]
    tab = (
        sub.groupby(["nome_disciplina", "faixa_risco"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Baixo", "Medio", "Alto"], fill_value=0)
    )
    tab_pct = tab.div(tab.sum(axis=1), axis=0) * 100
    tab_pct["n"] = tab.sum(axis=1)
    tab_pct = tab_pct.sort_values("Alto", ascending=False).head(n_show)
    plot_df = tab_pct[["Baixo", "Medio", "Alto"]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, n_show * 0.35)))
    plot_df.plot(kind="barh", stacked=True, ax=ax, color={"Baixo": "#2ecc71", "Medio": "#f1c40f", "Alto": "#e74c3c"})
    ax.set_xlabel("% das matriculas na disciplina")
    ax.legend(title="Faixa", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)
    plt.close(fig)


def secao_evolucao(d: pd.DataFrame) -> None:
    st.subheader("5. Evolucao do risco ao longo dos periodos")
    modo = st.radio("Agregacao", ["Por periodo letivo (calendario)", "Por semestre relativo no curso"], horizontal=True)
    if modo.startswith("Por periodo"):
        g = d.groupby("periodo_letivo", as_index=False).agg(risco_medio=("p_risco_insucesso", "mean"), n=("id_discente", "count"))
        g = g.merge(d.drop_duplicates(subset=["periodo_letivo"])[["periodo_letivo", "periodo_letivo_ord"]], on="periodo_letivo")
        g = g.sort_values("periodo_letivo_ord")
        x = "periodo_letivo"
        xl = "Periodo letivo (ano.periodo)"
    else:
        cap = st.slider("Truncar apos N semestres relativos", 5, 40, 20)
        g = d.groupby("semestre_rel_curso", as_index=False).agg(risco_medio=("p_risco_insucesso", "mean"), n=("id_discente", "count"))
        g = g[g["semestre_rel_curso"] <= cap]
        g = g.sort_values("semestre_rel_curso")
        x = "semestre_rel_curso"
        xl = "Semestre relativo (1 = primeira matricula do aluno no curso)"
    fig, ax = plt.subplots(figsize=(11, 4))
    xs = np.arange(len(g))
    ax.plot(xs, g["risco_medio"], marker="o", color="darkred")
    ax.fill_between(xs, g["risco_medio"], alpha=0.12, color="darkred")
    ax.set_xticks(xs)
    ax.set_xticklabels(g[x].astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Risco medio (prob. insucesso)")
    ax.set_xlabel(xl)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Cada ponto e a media de `p_risco_insucesso` sobre todas as matriculas naquele periodo.")


def secao_scatter_hist(d: pd.DataFrame, min_n: int) -> None:
    st.subheader("6. Risco previsto x historico de reprovacao (mesma area)")
    st.caption("Cada ponto = uma disciplina. Eixos sao medias por disciplina.")
    by_disc = (
        d.groupby("nome_disciplina")
        .agg(
            risco_medio=("p_risco_insucesso", "mean"),
            taxa_reprov_hist=("taxa_reprov_historica_antes", "mean"),
            n=("id_discente", "count"),
        )
        .query("n >= @min_n")
        .dropna(subset=["taxa_reprov_hist"])
    )
    if by_disc.empty:
        st.info("Sem pontos com historico calculado (denominador > 0).")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        by_disc["taxa_reprov_hist"],
        by_disc["risco_medio"],
        c=by_disc["n"],
        cmap="viridis",
        alpha=0.65,
        s=40,
    )
    plt.colorbar(sc, ax=ax, label="Matriculas")
    mx = max(by_disc["taxa_reprov_hist"].max(), by_disc["risco_medio"].max(), 0.05)
    ax.plot([0, mx], [0, mx], ls="--", color="gray", alpha=0.7, label="y = x")
    ax.set_xlabel("Taxa reprovacao historica antes (mesma area)")
    ax.set_ylabel("Risco medio previsto (modelo)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Disciplinas acima da diagonal: modelo atribui mais risco do que o historico simples na area sugere. "
        "Abaixo: historico mais severo que o score medio do modelo."
    )


def main() -> None:
    st.set_page_config(page_title="Risco academico por disciplina", page_icon="📊", layout="wide")
    st.title("Dashboard — previsao de risco por disciplina")
    st.markdown(
        "Foco: **onde** e **quando** o modelo indica maior probabilidade de insucesso, "
        "com cruzamento por periodo letivo (`ano.periodo`) e area de conhecimento."
    )

    if not BASE_CSV.exists() and not BASE_PARQUET.exists():
        st.error(
            f"Base nao encontrada (`{BASE_CSV}`). "
            "Execute na raiz: `python calcular_features_dashboard.py`"
        )
        st.stop()

    try:
        df = load_base()
    except Exception as e:
        st.error(f"Erro ao ler base: {e}")
        st.stop()

    d, min_n = sidebar_filtros(df)
    st.caption(
        f"Registros: **{len(d):,}** | Componentes no semestre = disciplinas distintas do aluno naquele `ano.periodo`."
    )

    secao_ranking(d, min_n)
    st.divider()
    secao_heatmap(d, min_n)
    st.divider()
    secao_distribuicao(d, min_n)
    st.divider()
    secao_concentracao_alto(d, min_n)
    st.divider()
    secao_evolucao(d)
    st.divider()
    secao_scatter_hist(d, min_n)


if __name__ == "__main__":
    main()
