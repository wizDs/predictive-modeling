import polars as pl
import streamlit as st

from load_data import Column, FeatureColumn


def render(joined_df: pl.DataFrame) -> None:
    st.header("Monthly Power Cost")

    relevant_data = joined_df.filter(pl.col(Column.HOURLY_TOTAL_COST).is_not_null())

    year_filter = st.selectbox(
        "Filter by Year",
        options=["All"]
        + sorted(
            relevant_data[FeatureColumn.YEAR].unique().to_list(), reverse=True
        ),
    )

    df_filtered = relevant_data
    if year_filter != "All":
        df_filtered = relevant_data.filter(
            pl.col(FeatureColumn.YEAR) == year_filter
        )

    monthly_cost = df_filtered.group_by(
        pl.col(FeatureColumn.MONTH_KEY), maintain_order=True
    ).agg(pl.col(Column.HOURLY_TOTAL_COST).sum().alias(Column.MONTHLY_TOTAL_COST))

    st.bar_chart(
        monthly_cost.to_pandas().set_index(FeatureColumn.MONTH_KEY)[
            Column.MONTHLY_TOTAL_COST
        ]
    )

    st.subheader("Cost by Time of Day")
    cost_by_event = (
        df_filtered.group_by(FeatureColumn.EVENT_OF_DAY)
        .agg(pl.col(Column.HOURLY_TOTAL_COST).sum().alias("total_cost"))
        .filter(pl.col("total_cost").is_not_null())
    )
    st.bar_chart(
        cost_by_event.to_pandas().set_index(FeatureColumn.EVENT_OF_DAY)[
            "total_cost"
        ]
    )
