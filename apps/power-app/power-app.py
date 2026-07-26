import datetime
import json
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from load_data import (
    Column,
    EnergyDataClient,
    MAX_WORKERS,
    _SPOT_FEE_DKK,
    join_prices_and_consumption_data,
    prices_monthly_df,
    split_request_params_all_versions,
)
from pages import (
    consumption_patterns,
    monthly_costs,
    overview,
    raw_data,
    weighted_avg_price,
    what_if,
)

PRICES_CACHE_PATH = Path(__file__).parent / ".prices_cache.parquet"
CONSUMPTION_CACHE_PATH = Path(__file__).parent / ".consumption_cache.parquet"
STATE_FILE = Path(__file__).parent / ".power_app_state.json"


def save_state(start: datetime.date, end: datetime.date) -> None:
    state = {"start": start.isoformat(), "end": end.isoformat()}
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_state() -> tuple[datetime.date, datetime.date]:
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
    client = EnergyDataClient()
    seq_params = split_request_params_all_versions(start, end)
    prices = client.get(seq_parameters=seq_params, n_jobs=MAX_WORKERS)
    return (
        pl.DataFrame(prices)
        .group_by(pl.col(Column.UTC_TIME).dt.truncate("1h"), maintain_order=True)
        .agg(pl.col(Column.SPOT_PRICE).mean().alias(Column.SPOT_PRICE))
        .with_columns(price_kwh_in_dkk=pl.col(Column.SPOT_PRICE) / 1000 + _SPOT_FEE_DKK)
    )


def load_prices(
    start: datetime.date, end: datetime.date, force_refresh: bool
) -> pl.DataFrame:
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
    if CONSUMPTION_CACHE_PATH.exists():
        if source is not None:
            CONSUMPTION_CACHE_PATH.unlink()
            st.cache_data.clear()
        else:
            return pl.read_parquet(CONSUMPTION_CACHE_PATH)
    try:
        consumption_df = pl.read_csv(
            source=source,
            decimal_comma=True,
            schema={"HourUTC": pl.Datetime, "SpotPriceDKK": pl.Float64},
        ).rename({"SpotPriceDKK": "consumption_kwh_hourly"})
    except Exception:
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

    st.sidebar.header("Configuration")

    consumption_file = st.sidebar.file_uploader(
        "Consumption CSV File",
        type=["csv"],
        help="Upload the consumption CSV file",
    )

    col1, col2 = st.sidebar.columns(2)
    start_date, end_date = load_state()
    with col1:
        start_date = st.date_input("Start Date", value=start_date)
    with col2:
        end_date = st.date_input("End Date", value=end_date)

    save_state(start_date, end_date)

    force_refresh = st.sidebar.button(
        "🔄 Refresh Prices", help="Force re-fetch prices from API"
    )

    if PRICES_CACHE_PATH.exists():
        st.sidebar.success("✅ Prices cached")
    else:
        st.sidebar.info("📡 Prices will be fetched")

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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Overview",
            "💰 Monthly Costs",
            "📈 Consumption Patterns",
            "⚖️ Weighted variable price",
            "🤔 What-if Analysis",
            "🔍 Raw Data",
        ]
    )

    with tab1:
        overview.render(joined_df)
    with tab2:
        monthly_costs.render(joined_df)
    with tab3:
        consumption_patterns.render(joined_df)
    with tab4:
        weighted_avg_price.render(joined_df, prices_monthly_df)
    with tab5:
        what_if.render(prices_df, consumption_df, joined_df)
    with tab6:
        raw_data.render(prices_df, consumption_df, joined_df)


if __name__ == "__main__":
    main()
