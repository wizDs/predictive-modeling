import datetime
import json
from pathlib import Path
from typing import Any
from matplotlib import pyplot as plt
import seaborn as sns
import polars as pl
import streamlit as st

from load_data import (
    Column,
    EnergyDataClient,
    FeatureColumn,
    MAX_WORKERS,
    _SPOT_FEE_DKK,
    cost_type_short_name,
    join_prices_and_consumption_data,
    prices_monthly_df,
    split_request_params_all_versions,
)

PRICES_CACHE_PATH = Path(__file__).parent / ".prices_cache.parquet"
CONSUMPTION_CACHE_PATH = Path(__file__).parent / ".consumption_cache.parquet"

STATE_FILE = Path(__file__).parent / ".power_app_state.json"


def save_state(start: datetime.date, end: datetime.date) -> None:
    """Save the payment interface state to a temp file."""
    state = {"start": start.isoformat(), "end": end.isoformat()}
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_state() -> tuple[datetime.date, datetime.date] | None:
    """Load the payment interface state from a temp file."""
    default_start = datetime.date(2020, 1, 1)
    default_end = datetime.date.today() - datetime.timedelta(days=1)
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        return (
            datetime.date.fromisoformat(state["start"]),
            datetime.date.fromisoformat(state["end"]),
        )
    return default_start, default_end


@st.cache_data
def fetch_prices(start: datetime.date, end: datetime.date) -> pl.DataFrame:
    """Fetch prices from API and cache them."""
    client = EnergyDataClient()
    seq_params = split_request_params_all_versions(start, end)
    prices = client.get(seq_parameters=seq_params, n_jobs=MAX_WORKERS)

    prices_df = (
        pl.DataFrame(prices)
        .group_by(pl.col(Column.UTC_TIME).dt.truncate("1h"), maintain_order=True)
        .agg(pl.col(Column.SPOT_PRICE).mean().alias(Column.SPOT_PRICE))
        .with_columns(price_kwh_in_dkk=pl.col(Column.SPOT_PRICE) / 1000 + _SPOT_FEE_DKK)
    )
    return prices_df


def load_prices(
    start: datetime.date, end: datetime.date, force_refresh: bool
) -> pl.DataFrame:
    """Load prices from cache or fetch from API."""
    if force_refresh and PRICES_CACHE_PATH.exists():
        PRICES_CACHE_PATH.unlink()
        st.cache_data.clear()

    if PRICES_CACHE_PATH.exists() and not force_refresh:
        return pl.read_parquet(PRICES_CACHE_PATH)

    with st.spinner("Fetching prices from API..."):
        prices_df = fetch_prices(start, end)
        prices_df.write_parquet(PRICES_CACHE_PATH)
        return prices_df


def load_consumption(source: Any) -> pl.DataFrame | None:
    """Load consumption data from CSV."""

    if CONSUMPTION_CACHE_PATH.exists():
        # if selected a new file, clear the cache
        if source is not None:
            CONSUMPTION_CACHE_PATH.unlink()
            st.cache_data.clear()
        # if no file is selected, load the cached data
        else:
            return pl.read_parquet(CONSUMPTION_CACHE_PATH)
    try:
        consumption_df = pl.read_csv(
            source=source,
            decimal_comma=True,
            schema={
                "HourUTC": pl.Datetime,
                "SpotPriceDKK": pl.Float64,
            },
        ).rename({"SpotPriceDKK": "consumption_kwh_hourly"})
    except Exception as _:
        consumption_df = (
            pl.read_csv(
                source=source,
                decimal_comma=True,
                separator=";",
                schema={
                    "MålepunktsID": pl.Int64,
                    "Fra_dato": pl.Utf8,
                    "Til_dato": pl.Utf8,
                    "Mængde": pl.Float64,
                    "Måleenhed": pl.Utf8,
                    "Kvalitet": pl.Utf8,
                    "Type": pl.Utf8,
                },
            )
            .with_columns(pl.col("Fra_dato").str.to_datetime("%d-%m-%Y %H:%M:%S"))
            .rename({"Fra_dato": "HourUTC", "Mængde": "consumption_kwh_hourly"})
        )
    consumption_df.write_parquet(CONSUMPTION_CACHE_PATH)
    return consumption_df


def main():
    st.set_page_config(page_title="Power Analysis", page_icon="⚡", layout="wide")
    st.title("⚡ Power Consumption Analysis")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Consumption data source
    consumption_file = st.sidebar.file_uploader(
        "Consumption CSV File",
        type=["csv"],
        help="Upload the consumption CSV file",
    )

    # Date range
    col1, col2 = st.sidebar.columns(2)
    start_date, end_date = load_state()
    with col1:
        start_date = st.date_input("Start Date", value=start_date)
    with col2:
        end_date = st.date_input("End Date", value=end_date)

    save_state(start_date, end_date)

    # Force refresh prices
    force_refresh = st.sidebar.button(
        "🔄 Refresh Prices", help="Force re-fetch prices from API"
    )

    # Show cache status
    if PRICES_CACHE_PATH.exists():
        st.sidebar.success("✅ Prices cached")
    else:
        st.sidebar.info("📡 Prices will be fetched")

    # Validate inputs
    if CONSUMPTION_CACHE_PATH.exists():
        st.sidebar.success("✅ Consumption cached")
    else:
        if consumption_file is None:
            st.sidebar.info("📡 Waiting for consumption file to be selected")
            return
        else:
            st.sidebar.info("📡 Consumption will be fetched from uploaded file")

    if start_date >= end_date:
        st.error("Start date must be before end date")
        return

    # Load data
    try:
        prices_df = load_prices(start_date, end_date, force_refresh)
        consumption_df = load_consumption(consumption_file)
        joined_df = join_prices_and_consumption_data(
            daily_prices_df=prices_df,
            daily_consumption_df=consumption_df,
            monthly_prices_df=prices_monthly_df,
        )
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "💰 Monthly Costs",
            "📈 Consumption Patterns",
            "🤔 What-if Analysis",
            "🔍 Raw Data",
        ]
    )

    with tab1:
        st.header("Price Overview")

        # Monthly average price over time
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

        # Summary metrics
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

    with tab2:
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

        # Cost by time of day
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

    with tab3:
        st.header("Consumption Patterns")

        # Hourly consumption pattern
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

        # Normalize per year so each year's distribution sums to 1
        st.subheader("Normalized Average Hourly Consumption by Year")
        year_columns = [
            c for c in hourly_pattern.columns if c != FeatureColumn.HOUR_OF_DAY
        ]
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

    with tab4:
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
        # Summary metrics
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

        # Summary metrics
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

    with tab5:
        st.header("Raw Data")

        st.subheader("Prices Data")
        st.dataframe(prices_df.head(100).to_pandas(), use_container_width=True)

        st.subheader("Consumption Data")
        st.dataframe(consumption_df.head(100).to_pandas(), use_container_width=True)

        st.subheader("Joined Data")
        st.dataframe(joined_df.head(100).to_pandas(), use_container_width=True)


if __name__ == "__main__":
    main()
