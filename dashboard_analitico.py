from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


DATA_PATH = Path("dados/saida_analise_risco_modelo.csv")


def _norm_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "Nao_informado", "None": "Nao_informado", "": "Nao_informado"})
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    rename_map = {
        "nome_componete_curricular": "nome_disciplina",
        "target": "aprovado_real",
        "pred": "aprovado_previsto",
    }
    df = df.rename(columns=rename_map)

    expected_cols = [
        "id_discente",
        "id_disciplina",
        "nome_disciplina",
        "sigla_departamento",
        "sigla_centro",
        "area_conhecimento",
        "forma_ingresso",
        "faixa_renda_familiar",
        "raca_declarada",
        "sexo",
        "idade",
        "primeiro_periodo",
        "aprovado_real",
        "aprovado_previsto",
        "p_sucesso",
        "p_risco_insucesso",
        "acerto",
        "faixa_risco",
        "faixa_idade",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    for col in [
        "nome_disciplina",
        "sigla_departamento",
        "sigla_centro",
        "area_conhecimento",
        "forma_ingresso",
        "faixa_renda_familiar",
        "raca_declarada",
        "sexo",
        "faixa_risco",
        "faixa_idade",
    ]:
        df[col] = _norm_text(df[col])

    df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    df["primeiro_periodo"] = pd.to_numeric(df["primeiro_periodo"], errors="coerce").fillna(0).astype(int)
    df["aprovado_real"] = pd.to_numeric(df["aprovado_real"], errors="coerce").fillna(0).astype(int)
    df["aprovado_previsto"] = pd.to_numeric(df["aprovado_previsto"], errors="coerce").fillna(0).astype(int)
    df["p_risco_insucesso"] = pd.to_numeric(df["p_risco_insucesso"], errors="coerce").fillna(0.0)
    df["p_sucesso"] = pd.to_numeric(df["p_sucesso"], errors="coerce").fillna(0.0)

    # Proxy de cancelamento: reprovado com risco alto do modelo.
    df["indicador_cancelamento_risco"] = np.where(
        (df["aprovado_real"] == 0) & (df["p_risco_insucesso"] >= 0.7), "Potencial cancelamento", "Sem indicio forte"
    )
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")
    centros = st.sidebar.multiselect("Centro", sorted(df["sigla_centro"].unique()))
    areas = st.sidebar.multiselect("Area de conhecimento", sorted(df["area_conhecimento"].unique()))
    riscos = st.sidebar.multiselect("Faixa de risco", sorted(df["faixa_risco"].unique()))
    primeiro_periodo = st.sidebar.selectbox("Recorte por periodo", ["Todos", "Primeiro periodo", "Periodos seguintes"])

    filtered = df.copy()
    if centros:
        filtered = filtered[filtered["sigla_centro"].isin(centros)]
    if areas:
        filtered = filtered[filtered["area_conhecimento"].isin(areas)]
    if riscos:
        filtered = filtered[filtered["faixa_risco"].isin(riscos)]
    if primeiro_periodo == "Primeiro periodo":
        filtered = filtered[filtered["primeiro_periodo"] == 1]
    elif primeiro_periodo == "Periodos seguintes":
        filtered = filtered[filtered["primeiro_periodo"] == 0]
    return filtered


def metricas_gerais(df: pd.DataFrame) -> None:
    total = len(df)
    taxa_aprovacao = (df["aprovado_real"].mean() * 100.0) if total else 0.0
    taxa_reprovacao = 100.0 - taxa_aprovacao
    risco_medio = (df["p_risco_insucesso"].mean() * 100.0) if total else 0.0
    alto_risco = ((df["p_risco_insucesso"] >= 0.7).mean() * 100.0) if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{total:,}".replace(",", "."))
    c2.metric("Taxa de aprovacao", f"{taxa_aprovacao:.1f}%")
    c3.metric("Taxa de reprovacao", f"{taxa_reprovacao:.1f}%")
    c4.metric("Alto risco (>= 70%)", f"{alto_risco:.1f}%")
    st.caption(f"Risco medio de insucesso: {risco_medio:.1f}%")


def tabela_disciplinas_criticas(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("nome_disciplina", as_index=False)
        .agg(
            total_registros=("id_discente", "count"),
            taxa_reprovacao=("aprovado_real", lambda x: (1 - x.mean()) * 100),
            risco_medio=("p_risco_insucesso", lambda x: x.mean() * 100),
            alto_risco=("p_risco_insucesso", lambda x: (x >= 0.7).mean() * 100),
        )
        .sort_values(["taxa_reprovacao", "risco_medio", "total_registros"], ascending=[False, False, False])
    )
    return g


def area_por_periodo(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["periodo_grupo"] = np.where(base["primeiro_periodo"] == 1, "Primeiro periodo", "Periodos seguintes")
    area = (
        base.groupby(["periodo_grupo", "area_conhecimento"], as_index=False)
        .agg(
            total=("id_discente", "count"),
            taxa_reprovacao=("aprovado_real", lambda x: (1 - x.mean()) * 100),
            risco_medio=("p_risco_insucesso", lambda x: x.mean() * 100),
        )
        .sort_values(["periodo_grupo", "taxa_reprovacao"], ascending=[True, False])
    )
    return area


def perfil(df: pd.DataFrame, aprovado: int | None) -> dict[str, pd.DataFrame]:
    subset = df.copy() if aprovado is None else df[df["aprovado_real"] == aprovado]
    if subset.empty:
        return {"sexo": pd.DataFrame(), "raca": pd.DataFrame(), "renda": pd.DataFrame(), "ingresso": pd.DataFrame()}

    def dist(col: str) -> pd.DataFrame:
        return (
            subset[col]
            .value_counts(normalize=True)
            .rename("proporcao")
            .mul(100)
            .round(2)
            .reset_index()
            .rename(columns={"index": col})
        )

    return {
        "sexo": dist("sexo"),
        "raca": dist("raca_declarada"),
        "renda": dist("faixa_renda_familiar"),
        "ingresso": dist("forma_ingresso"),
    }


def main() -> None:
    st.set_page_config(page_title="Dashboard Analitico de Risco Academico", page_icon=":bar_chart:", layout="wide")
    st.title("Dashboard Analitico de Aprovações, Reprovações e Risco")
    st.write(
        "Painel para apoio a coordenacao de curso com foco em disciplinas criticas, "
        "areas de maior falha, risco de insucesso e perfil dos estudantes."
    )

    if not DATA_PATH.exists():
        st.error(f"Arquivo nao encontrado: {DATA_PATH}")
        st.stop()

    df = load_data()
    filtered = apply_filters(df)
    metricas_gerais(filtered)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Disciplinas criticas",
            "Areas por periodo",
            "Cancelamento e risco",
            "Perfil dos estudantes",
            "Matriz de confusao e risco",
        ]
    )

    with tab1:
        st.subheader("Quais disciplinas mais retem e concentram risco?")
        top_n = st.slider("Top disciplinas", min_value=5, max_value=40, value=15, step=5)
        minimo_registros = st.number_input("Minimo de registros por disciplina", min_value=1, max_value=500, value=20)
        criticas = tabela_disciplinas_criticas(filtered)
        criticas = criticas[criticas["total_registros"] >= minimo_registros].head(top_n)
        st.dataframe(
            criticas.style.format(
                {
                    "taxa_reprovacao": "{:.1f}%",
                    "risco_medio": "{:.1f}%",
                    "alto_risco": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )
        st.bar_chart(criticas.set_index("nome_disciplina")["taxa_reprovacao"])

    with tab2:
        st.subheader("Onde os alunos falham no primeiro periodo e nos seguintes?")
        area = area_por_periodo(filtered)
        st.dataframe(area, use_container_width=True)
        pivot = area.pivot(index="area_conhecimento", columns="periodo_grupo", values="taxa_reprovacao").fillna(0)
        st.bar_chart(pivot)

    with tab3:
        st.subheader("Disciplinas com maior impacto potencial em cancelamento")
        cancel = (
            filtered[filtered["indicador_cancelamento_risco"] == "Potencial cancelamento"]
            .groupby("nome_disciplina", as_index=False)
            .agg(
                casos=("id_discente", "count"),
                taxa_reprovacao=("aprovado_real", lambda x: (1 - x.mean()) * 100),
                risco_medio=("p_risco_insucesso", lambda x: x.mean() * 100),
            )
            .sort_values(["casos", "risco_medio"], ascending=[False, False])
            .head(20)
        )
        st.dataframe(cancel, use_container_width=True)
        if not cancel.empty:
            st.bar_chart(cancel.set_index("nome_disciplina")["casos"])
        st.caption(
            "Observacao: 'Potencial cancelamento' e um indicador analitico (proxy), "
            "baseado em reprovacao real com probabilidade de insucesso >= 70%."
        )

    with tab4:
        st.subheader("Qual o perfil do aluno do curso, de quem reprova e de quem passa?")
        op = st.radio("Selecione o grupo", ["Perfil geral", "Aprovados", "Reprovados"], horizontal=True)
        target_map = {"Perfil geral": None, "Aprovados": 1, "Reprovados": 0}
        dist = perfil(filtered, target_map[op])

        c1, c2 = st.columns(2)
        c1.write("Sexo")
        c1.dataframe(dist["sexo"], use_container_width=True)
        c2.write("Raca declarada")
        c2.dataframe(dist["raca"], use_container_width=True)

        c3, c4 = st.columns(2)
        c3.write("Faixa de renda familiar")
        c3.dataframe(dist["renda"], use_container_width=True)
        c4.write("Forma de ingresso")
        c4.dataframe(dist["ingresso"], use_container_width=True)

        st.write("Distribuicao de idades")
        idade = filtered if target_map[op] is None else filtered[filtered["aprovado_real"] == target_map[op]]
        if not idade.empty:
            hist = np.histogram(idade["idade"].dropna(), bins=10)
            hist_df = pd.DataFrame({"faixa": range(1, len(hist[0]) + 1), "quantidade": hist[0]})
            st.bar_chart(hist_df.set_index("faixa")["quantidade"])

    with tab5:
        st.subheader("Qualidade do modelo e distribuicao de risco")
        cm = (
            filtered.groupby(["aprovado_real", "aprovado_previsto"])
            .size()
            .reset_index(name="qtd")
            .pivot(index="aprovado_real", columns="aprovado_previsto", values="qtd")
            .fillna(0)
            .astype(int)
        )
        cm.index = ["Real: Reprovado", "Real: Aprovado"] if len(cm.index) == 2 else cm.index
        cm.columns = ["Previsto: Reprovado", "Previsto: Aprovado"] if len(cm.columns) == 2 else cm.columns
        st.dataframe(cm, use_container_width=True)

        risco = (
            filtered.assign(status=np.where(filtered["aprovado_real"] == 1, "Aprovado", "Reprovado"))
            .groupby(["faixa_risco", "status"], as_index=False)
            .size()
            .rename(columns={"size": "quantidade"})
        )
        st.dataframe(risco, use_container_width=True)
        risco_pivot = risco.pivot(index="faixa_risco", columns="status", values="quantidade").fillna(0)
        st.bar_chart(risco_pivot)


if __name__ == "__main__":
    main()
