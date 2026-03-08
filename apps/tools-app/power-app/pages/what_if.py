import polars as pl
import streamlit as st

from load_data import Column, FeatureColumn, join_prices_and_consumption_data, prices_monthly_df


def render(
    prices_df: pl.DataFrame,
    consumption_df: pl.DataFrame,
    joined_df: pl.DataFrame,
) -> None:
    st.header("What-if Analysis")

    offset_years = st.number_input(
        "Offset by (years)", value=0, min_value=0, max_value=10
    )

    what_if_df = join_prices_and_consumption_data(
        daily_prices_df=prices_df,
        daily_consumption_df=consumption_df.with_columns(
            pl.col(Column.UTC_TIME).dt.offset_by(f"-{offset_years}y")
        ),
        monthly_prices_df=prices_monthly_df,
    )
    what_if_df = (
        what_if_df.filter(pl.col(Column.HOURLY_CONSUMPTION).is_not_null())
        .sort(Column.UTC_TIME)
        .with_columns(
            pl.col(Column.HOURLY_CONSUMPTION)
            .rolling_sum_by(Column.UTC_TIME, window_size="1mo")
            .alias(Column.HOURLY_CONSUMPTION_ROLLING_SUM)
        )
        .with_columns(
            (
                pl.col(Column.FIXED_HOURLY_TOTAL_COST)
                - pl.col(Column.HOURLY_TOTAL_COST)
            ).alias("cost_difference")
        )
    )

    col1, col2, col3 = st.columns(3)
    aligned_df = what_if_df.filter(
        pl.col(Column.FIXED_HOURLY_TOTAL_COST).is_not_null()
    ).filter(pl.col(Column.HOURLY_TOTAL_COST).is_not_null())
    with col1:
        total_fixed_price = aligned_df[Column.FIXED_HOURLY_TOTAL_COST].sum()
        st.metric(
            "Total Fixed Price (DKK)",
            f"{total_fixed_price:,.0f}" if total_fixed_price else "N/A",
        )
    with col2:
        total_variable_price = aligned_df[Column.HOURLY_TOTAL_COST].sum()
        st.metric(
            "Total Variable Price (DKK)",
            f"{total_variable_price:.0f}" if total_variable_price else "N/A",
        )
    with col3:
        total_cost_difference = total_fixed_price - total_variable_price
        st.metric(
            "Total Cost Difference (DKK)",
            f"{total_cost_difference:,.0f}" if total_cost_difference else "N/A",
        )

    col4, col5, col6 = st.columns(3)
    month_count = aligned_df[FeatureColumn.MONTH_KEY].n_unique()
    year_count = month_count / 12
    with col4:
        fixed_price = aligned_df[Column.FIXED_HOURLY_TOTAL_COST].sum() / year_count
        st.metric(
            "Fixed Price (DKK/year)",
            f"{fixed_price:,.0f}" if fixed_price else "N/A",
        )
    with col5:
        variable_price = aligned_df[Column.HOURLY_TOTAL_COST].sum() / year_count
        st.metric(
            "Variable Price (DKK/year)",
            f"{variable_price:.0f}" if variable_price else "N/A",
        )
    with col6:
        cost_difference = fixed_price - variable_price
        st.metric(
            "Cost Difference (DKK/year)",
            f"{cost_difference:,.0f}" if cost_difference else "N/A",
        )

    st.markdown("")

    df = (
        what_if_df.group_by(pl.col(FeatureColumn.MONTH_KEY), maintain_order=True)
        .agg(
            [
                pl.col(Column.HOURLY_TOTAL_COST)
                .sum()
                .alias(Column.MONTHLY_TOTAL_COST),
                pl.col(Column.FIXED_HOURLY_TOTAL_COST)
                .sum()
                .alias(Column.FIXED_MONTHLY_TOTAL_COST),
            ]
        )
        .filter(pl.col(Column.MONTHLY_TOTAL_COST) > 0)
    )
    st.bar_chart(
        df,
        x=FeatureColumn.MONTH_KEY,
        y=Column.MONTHLY_TOTAL_COST,
        height=500,
    )

    st.subheader("Variable vs Fixed Price")
    compare_df = (
        df.filter(pl.col(Column.FIXED_MONTHLY_TOTAL_COST) > 0)
        .rename(
            {
                Column.MONTHLY_TOTAL_COST: "variable",
                Column.FIXED_MONTHLY_TOTAL_COST: "fixed",
            }
        )
        .select(FeatureColumn.MONTH_KEY, "variable", "fixed")
        .sort(FeatureColumn.MONTH_KEY)
    )
    st.bar_chart(
        compare_df,
        x=FeatureColumn.MONTH_KEY,
        y=["variable", "fixed"],
        stack=False,
        height=500,
        use_container_width=True,
    )

    with st.expander("Investigate month"):
        _month = st.multiselect(
            "Select Month",
            options=compare_df[FeatureColumn.MONTH_KEY].unique().to_list(),
        )
        investigation_df = what_if_df.filter(
            pl.col(FeatureColumn.MONTH_KEY).is_in(_month)
        )
        st.subheader("Cost Difference (fixed - variable)")
        st.bar_chart(
            investigation_df,
            x=Column.TIMESTAMP,
            y="cost_difference",
            y_label="Cost Difference (DKK)",
            stack=False,
            height=500,
            use_container_width=True,
        )
        st.bar_chart(
            data=investigation_df,
            x=Column.TIMESTAMP,
            y=Column.HOURLY_PRICE,
            stack=False,
            height=500,
            use_container_width=True,
        )

    with st.expander("Extreme distributions"):
        cost_parameter = st.number_input(
            "Expensive parameter",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )

        daily_consumption = (
            aligned_df.group_by(FeatureColumn.MONTH_KEY, FeatureColumn.DAY_KEY)
            .agg(pl.sum(Column.HOURLY_CONSUMPTION).alias("daily_consumption"))
            .sort(FeatureColumn.DAY_KEY)
        )

        expensive_pattern = pl.DataFrame(
            {
                FeatureColumn.HOUR_OF_DAY: [17, 18, 19, 20],
                "expensive_pct_of_total_consumption": [0.25, 0.25, 0.25, 0.25],
            }
        )
        cheap_pattern = pl.DataFrame(
            {
                FeatureColumn.HOUR_OF_DAY: [1, 2, 3, 4],
                "cheap_pct_of_total_consumption": [0.25, 0.25, 0.25, 0.25],
            }
        )
        hourly_pattern = (
            aligned_df.group_by(FeatureColumn.HOUR_OF_DAY)
            .agg(pl.sum(Column.HOURLY_CONSUMPTION).alias("hourly_consumption"))
            .with_columns(
                (pl.col("hourly_consumption") / pl.sum("hourly_consumption")).alias(
                    "pct_of_total_consumption"
                )
            )
            .sort(FeatureColumn.HOUR_OF_DAY)
            .join(expensive_pattern, on=FeatureColumn.HOUR_OF_DAY, how="left")
            .join(cheap_pattern, on=FeatureColumn.HOUR_OF_DAY, how="left")
            .fill_null(0)
        )

        expensive_parameter = (
            1 - (1 - cost_parameter) / 0.5 if cost_parameter > 0.5 else 0
        )
        cheap_parameter = 1 - cost_parameter / 0.5 if cost_parameter < 0.5 else 0
        balanced_parameter = 1 - max(expensive_parameter, cheap_parameter)
        if expensive_parameter + cheap_parameter + balanced_parameter != 1:
            st.error(
                "Expensive parameter, cheap parameter and balanced parameter must sum to 1"
            )
            return

        hourly_consumption = daily_consumption.join(
            hourly_pattern, how="cross"
        ).with_columns(
            (
                pl.col("daily_consumption")
                * (
                    balanced_parameter * pl.col("pct_of_total_consumption")
                    + expensive_parameter
                    * pl.col("expensive_pct_of_total_consumption")
                    + cheap_parameter * pl.col("cheap_pct_of_total_consumption")
                )
            ).alias("smoothened_consumption")
        )
        st.bar_chart(
            data=hourly_consumption.group_by(FeatureColumn.HOUR_OF_DAY).agg(
                pl.sum("smoothened_consumption")
            ),
            x=FeatureColumn.HOUR_OF_DAY,
            y="smoothened_consumption",
            y_label="Consumption (kWh)",
            height=500,
            use_container_width=True,
        )
        simulated_cost = (
            hourly_consumption.join(
                joined_df.select(
                    FeatureColumn.DAY_KEY,
                    FeatureColumn.HOUR_OF_DAY,
                    Column.HOURLY_PRICE,
                ),
                on=[FeatureColumn.DAY_KEY, FeatureColumn.HOUR_OF_DAY],
                how="left",
            )
            .with_columns(
                (
                    pl.col(Column.HOURLY_PRICE) * pl.col("smoothened_consumption")
                ).alias(Column.HOURLY_TOTAL_COST)
            )
            .select(pl.sum(Column.HOURLY_TOTAL_COST))
            .item()
        )
        st.markdown(f"Simulated total cost: {simulated_cost:,.0f} DKK")
