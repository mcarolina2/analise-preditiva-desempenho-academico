"""Carregamento em cache da base do dashboard (compartilhado entre paginas Streamlit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

BASE_CSV = Path("dados/base_dashboard_risco.csv")
BASE_PARQUET = Path("dados/base_dashboard_risco.parquet")


@st.cache_data(show_spinner=False)
def load_base() -> pd.DataFrame:
    if BASE_PARQUET.exists():
        try:
            return pd.read_parquet(BASE_PARQUET, engine="fastparquet")
        except Exception:
            pass
    if BASE_CSV.exists():
        return pd.read_csv(BASE_CSV, low_memory=False)
    raise FileNotFoundError(f"Nem {BASE_PARQUET} nem {BASE_CSV} encontrados.")
