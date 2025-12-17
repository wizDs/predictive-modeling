import datetime
from pathlib import Path

import polars as pl
import streamlit as st

from load_data import (
    Column,
    EnergyDataClient,
    FeatureColumn,
    MAX_WORKERS,
    _SPOT_FEE_DKK,
    join_prices_and_consumption_data,
    prices_monthly_df,
    split_request_params_all_versions,
)

PRICES_CACHE_PATH = Path(__file__).parent / ".prices_cache.parquet"
DEFAULT_CONSUMPTION_PATH = "/Users/wiz/projects/predictive-modeling/data/energi-data.csv"


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


def load_prices(start: datetime.date, end: datetime.date, force_refresh: bool) -> pl.DataFrame:
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


def load_consumption(csv_path: str) -> pl.DataFrame:
    """Load consumption data from CSV."""
    return pl.read_csv(
        source=csv_path,
        decimal_comma=True,
        schema={
            "HourUTC": pl.Datetime,
            "SpotPriceDKK": pl.Float64,
        },
    ).rename({"SpotPriceDKK": "consumption_kwh_hourly"})


def main():
    st.set_page_config(page_title="Power Analysis", page_icon="⚡", layout="wide")
    st.title("⚡ Power Consumption Analysis")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Consumption data source
    consumption_path = st.sidebar.text_input(
        "Consumption CSV Path",
        value=DEFAULT_CONSUMPTION_PATH,
        help="Path to the consumption CSV file",
    )

    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.date.today())

    # Force refresh prices
    force_refresh = st.sidebar.button("🔄 Refresh Prices", help="Force re-fetch prices from API")

    # Show cache status
    if PRICES_CACHE_PATH.exists():
        st.sidebar.success("✅ Prices cached")
    else:
        st.sidebar.info("📡 Prices will be fetched")

    # Validate inputs
    if not Path(consumption_path).exists():
        st.error(f"Consumption file not found: {consumption_path}")
        return

    if start_date >= end_date:
        st.error("Start date must be before end date")
        return

    # Load data
    try:
        prices_df = load_prices(start_date, end_date, force_refresh)
        consumption_df = load_consumption(consumption_path)
        joined_df = join_prices_and_consumption_data(
            daily_prices_df=prices_df,
            daily_consumption_df=consumption_df,
            monthly_prices_df=prices_monthly_df,
        )
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "💰 Monthly Costs", "📈 Consumption Patterns", "🔍 Raw Data"]
    )

    with tab1:
        st.header("Price Overview")

        # Monthly average price over time
        monthly_prices = (
            joined_df.group_by(pl.col(Column.TIMESTAMP).dt.truncate("1mo"), maintain_order=True)
            .agg(pl.col(Column.HOURLY_PRICE).mean().alias(Column.MONTHLY_PRICE_AVG))
            .with_columns(pl.col(Column.MONTHLY_PRICE_AVG).rolling_mean(window_size=3).alias("rolling_avg"))
        )
        st.subheader("Monthly Average Price (DKK/kWh)")
        st.line_chart(
            monthly_prices.to_pandas().set_index(Column.TIMESTAMP)[[Column.MONTHLY_PRICE_AVG, "rolling_avg"]]
        )

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_price = joined_df[Column.HOURLY_PRICE].mean()
            st.metric("Avg Price (DKK/kWh)", f"{avg_price:.3f}" if avg_price else "N/A")
        with col2:
            total_consumption = joined_df[Column.HOURLY_CONSUMPTION].sum()
            st.metric("Total Consumption (kWh)", f"{total_consumption:,.0f}" if total_consumption else "N/A")
        with col3:
            total_cost = joined_df[Column.HOURLY_TOTAL_COST].sum()
            st.metric("Total Cost (DKK)", f"{total_cost:,.0f}" if total_cost else "N/A")

    with tab2:
        st.header("Monthly Power Cost")

        year_filter = st.selectbox(
            "Filter by Year",
            options=["All"] + sorted(joined_df[FeatureColumn.YEAR].unique().to_list(), reverse=True),
        )

        df_filtered = joined_df
        if year_filter != "All":
            df_filtered = joined_df.filter(pl.col(FeatureColumn.YEAR) == year_filter)

        monthly_cost = (
            df_filtered.group_by(pl.col(Column.TIMESTAMP).dt.truncate("1mo"), maintain_order=True)
            .agg(pl.col(Column.HOURLY_TOTAL_COST).sum().alias(Column.MONTHLY_TOTAL_COST))
            .filter(pl.col(Column.MONTHLY_TOTAL_COST).is_not_null())
        )

        st.bar_chart(monthly_cost.to_pandas().set_index(Column.TIMESTAMP)[Column.MONTHLY_TOTAL_COST])

        # Cost by time of day
        st.subheader("Cost by Time of Day")
        cost_by_event = (
            df_filtered.group_by(FeatureColumn.EVENT_OF_DAY)
            .agg(pl.col(Column.HOURLY_TOTAL_COST).sum().alias("total_cost"))
            .filter(pl.col("total_cost").is_not_null())
        )
        st.bar_chart(cost_by_event.to_pandas().set_index(FeatureColumn.EVENT_OF_DAY)["total_cost"])

    with tab3:
        st.header("Consumption Patterns")

        # Hourly consumption pattern
        st.subheader("Average Hourly Consumption by Year")
        hourly_pattern = (
            joined_df.filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
            .group_by(FeatureColumn.YEAR, FeatureColumn.HOUR_OF_DAY)
            .agg(pl.mean(Column.HOURLY_CONSUMPTION).alias("avg_consumption"))
            .sort(FeatureColumn.HOUR_OF_DAY)
            .pivot(on=FeatureColumn.YEAR, index=FeatureColumn.HOUR_OF_DAY, values="avg_consumption")
        )
        st.line_chart(hourly_pattern.to_pandas().set_index(FeatureColumn.HOUR_OF_DAY))

        # Monthly heatmap data
        st.subheader("Consumption Heatmap (Month vs Hour)")
        heatmap_data = (
            joined_df.filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
            .group_by(FeatureColumn.MONTH, FeatureColumn.HOUR_OF_DAY)
            .agg(pl.mean(Column.HOURLY_CONSUMPTION).alias("avg_consumption"))
            .sort(FeatureColumn.MONTH, FeatureColumn.HOUR_OF_DAY)
            .pivot(
                on=FeatureColumn.HOUR_OF_DAY,
                index=FeatureColumn.MONTH,
                values="avg_consumption",
            )
        )
        st.dataframe(heatmap_data.to_pandas().set_index(FeatureColumn.MONTH), use_container_width=True)

    with tab4:
        st.header("Raw Data")

        st.subheader("Prices Data")
        st.dataframe(prices_df.head(100).to_pandas(), use_container_width=True)

        st.subheader("Consumption Data")
        st.dataframe(consumption_df.head(100).to_pandas(), use_container_width=True)

        st.subheader("Joined Data")
        st.dataframe(joined_df.head(100).to_pandas(), use_container_width=True)


if __name__ == "__main__":
    main()
