import polars as pl
import streamlit as st

from load_data import Column


def render(joined_df: pl.DataFrame) -> None:
    st.header("Price Overview")

    monthly_prices = (
        joined_df.group_by(
            pl.col(Column.TIMESTAMP).dt.truncate("1mo"), maintain_order=True
        )
        .agg(pl.col(Column.HOURLY_PRICE).mean().alias(Column.MONTHLY_PRICE_AVG))
        .with_columns(
            pl.col(Column.MONTHLY_PRICE_AVG)
            .rolling_mean(window_size=3)
            .alias("rolling_avg")
        )
    )
    st.subheader("Monthly Average Price (DKK/kWh)")
    st.line_chart(
        monthly_prices.to_pandas().set_index(Column.TIMESTAMP)[
            [Column.MONTHLY_PRICE_AVG, "rolling_avg"]
        ]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_price = joined_df[Column.HOURLY_PRICE].mean()
        st.metric("Avg Price (DKK/kWh)", f"{avg_price:.3f}" if avg_price else "N/A")
    with col2:
        total_consumption = joined_df[Column.HOURLY_CONSUMPTION].sum()
        st.metric(
            "Total Consumption (kWh)",
            f"{total_consumption:,.0f}" if total_consumption else "N/A",
        )
    with col3:
        total_cost = joined_df[Column.HOURLY_TOTAL_COST].sum()
        st.metric("Total Cost (DKK)", f"{total_cost:,.0f}" if total_cost else "N/A")
