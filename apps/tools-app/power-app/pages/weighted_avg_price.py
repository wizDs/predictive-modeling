import altair as alt
import polars as pl
import streamlit as st

from load_data import Column, FeatureColumn


def render(joined_df: pl.DataFrame, prices_monthly_df: pl.DataFrame) -> None:
    st.header("Weighted avg price")

    relevant_data = joined_df.filter(pl.col(Column.HOURLY_TOTAL_COST).is_not_null())

    year_filter = st.selectbox(
        "Filter by Year",
        options=["All"]
        + sorted(relevant_data[FeatureColumn.YEAR].unique().to_list(), reverse=True),
        key="weighted_avg_price_year_filter",
    )

    df_filtered = relevant_data
    if year_filter != "All":
        df_filtered = relevant_data.filter(pl.col(FeatureColumn.YEAR) == year_filter)

    monthly_cost = (
        df_filtered.group_by(
            pl.col(FeatureColumn.MONTH_KEY),
            pl.col(FeatureColumn.YEAR),
            pl.col(FeatureColumn.MONTH),
            maintain_order=True,
        )
        .agg(
            pl.col(Column.HOURLY_TOTAL_COST).sum().alias(Column.MONTHLY_TOTAL_COST),
            pl.col(Column.HOURLY_CONSUMPTION).sum().alias(Column.MONTHLY_CONSUMPTION),
        )
        .with_columns(
            (
                pl.col(Column.MONTHLY_TOTAL_COST) / pl.col(Column.MONTHLY_CONSUMPTION)
            ).alias("weighted_cost_kwh")
        )
        .join(
            prices_monthly_df.select(
                FeatureColumn.YEAR,
                FeatureColumn.MONTH,
                Column.FIXED_HOURLY_PRICE,
            ),
            on=[FeatureColumn.YEAR, FeatureColumn.MONTH],
            how="left",
        )
        .sort(FeatureColumn.MONTH_KEY)
    )

    df_plot = (
        monthly_cost.select(
            FeatureColumn.MONTH_KEY, "weighted_cost_kwh", Column.FIXED_HOURLY_PRICE
        )
        .to_pandas()
        .melt(id_vars=FeatureColumn.MONTH_KEY, var_name="type", value_name="price")
    )

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X(f"{FeatureColumn.MONTH_KEY}:O", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("price:Q", title="DKK/kWh"),
            color="type:N",
            xOffset="type:N",
        )
        .properties(width="container")
    )
    st.altair_chart(chart, use_container_width=True)
