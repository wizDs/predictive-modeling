import polars as pl
import streamlit as st

from load_data import Column, FeatureColumn


def render(joined_df: pl.DataFrame) -> None:
    st.header("Consumption Patterns")

    st.subheader("Average Hourly Consumption by Year")
    hourly_pattern = (
        joined_df.filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
        .group_by(FeatureColumn.YEAR, FeatureColumn.HOUR_OF_DAY)
        .agg(pl.mean(Column.HOURLY_CONSUMPTION).alias("avg_consumption"))
        .sort(FeatureColumn.HOUR_OF_DAY)
        .pivot(
            on=FeatureColumn.YEAR,
            index=FeatureColumn.HOUR_OF_DAY,
            values="avg_consumption",
        )
    )
    st.line_chart(
        data=hourly_pattern,
        x=FeatureColumn.HOUR_OF_DAY,
        y_label="Consumption (kWh)",
        height=500,
        use_container_width=True,
    )

    st.subheader("Normalized Average Hourly Consumption by Year")
    year_columns = [c for c in hourly_pattern.columns if c != FeatureColumn.HOUR_OF_DAY]
    normalized_hourly_pattern = hourly_pattern.select(
        FeatureColumn.HOUR_OF_DAY,
        *[
            (pl.col(year) / pl.col(year).sum() * 100).alias(year)
            for year in year_columns
        ],
    )
    st.line_chart(
        data=normalized_hourly_pattern,
        x=FeatureColumn.HOUR_OF_DAY,
        y=year_columns,
        y_label="Normalized Consumption (%)",
        height=500,
        use_container_width=True,
    )
