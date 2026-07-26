import polars as pl
import streamlit as st


def render(
    prices_df: pl.DataFrame,
    consumption_df: pl.DataFrame,
    joined_df: pl.DataFrame,
) -> None:
    st.header("Raw Data")

    st.subheader("Prices Data")
    st.dataframe(prices_df.head(100).to_pandas(), use_container_width=True)

    st.subheader("Consumption Data")
    st.dataframe(consumption_df.head(100).to_pandas(), use_container_width=True)

    st.subheader("Joined Data")
    st.dataframe(joined_df.head(100).to_pandas(), use_container_width=True)
