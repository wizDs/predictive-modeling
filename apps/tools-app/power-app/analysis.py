# %%
import datetime
import polars as pl
from load_data import PricesMonthly
from load_data import EnergyDataClient
from load_data import split_request_params_all_versions
from load_data import join_prices_and_consumption_data
from load_data import Column
from load_data import FeatureColumn
from load_data import prices_monthly_ts
from load_data import prices_monthly_df
from collections.abc import Sequence
from load_data import Elspotprices
from load_data import MAX_WORKERS
from load_data import _SPOT_FEE_DKK
import seaborn as sns
import matplotlib.pyplot as plt


# %%
start = datetime.date(2020, 1, 1)
end = datetime.date(2025, 12, 4)

# %%
client = EnergyDataClient()

# %%
seq_params = split_request_params_all_versions(start, end)
prices: Sequence[Elspotprices] = client.get(
    seq_parameters=seq_params, n_jobs=MAX_WORKERS
)

prices_df: pl.DataFrame = (
    pl.DataFrame(prices)
    .group_by(pl.col(Column.UTC_TIME).dt.truncate("1h"), maintain_order=True)
    .agg(pl.col(Column.SPOT_PRICE).mean().alias(Column.SPOT_PRICE))
    .with_columns(price_kwh_in_dkk=pl.col(Column.SPOT_PRICE) / 1000 + _SPOT_FEE_DKK)
)


# %%
# Load consumption data
consumption_df: pl.DataFrame = pl.read_csv(
    source="/Users/wiz/projects/predictive-modeling/data/energi-data.csv",
    decimal_comma=True,
    schema={
        "HourUTC": pl.Datetime,
        "SpotPriceDKK": pl.Float64,
    },
).rename({"SpotPriceDKK": "consumption_kwh_hourly"})


# %% [markdown]
# # Joined

# %%
joined_df = join_prices_and_consumption_data(
    daily_prices_df=prices_df,
    daily_consumption_df=consumption_df,
    monthly_prices_df=prices_monthly_df,
)

# %%
sns.set_theme(style="whitegrid")

# %% [markdown]
# # Prices monthly avg over time

# %%
df = (
    joined_df.group_by(pl.col(Column.TIMESTAMP).dt.truncate("1mo"), maintain_order=True)
    .agg(pl.col(Column.HOURLY_PRICE).mean().alias(Column.MONTHLY_PRICE_AVG))
    .with_columns(pl.col(Column.MONTHLY_PRICE_AVG).rolling_mean(window_size=3))
)
sns.lineplot(data=df, x=Column.TIMESTAMP, y=Column.MONTHLY_PRICE_AVG)

# %% [markdown]
# # Monthly Power Cost

# %%
df = (
    joined_df.filter(pl.col(Column.TIMESTAMP).dt.year() >= 2024)
    .group_by(pl.col(Column.TIMESTAMP).dt.truncate("1mo"), maintain_order=True)
    .agg(pl.col(Column.HOURLY_TOTAL_COST).sum().alias(Column.MONTHLY_TOTAL_COST))
)
sns.barplot(data=df, x=Column.TIMESTAMP, y=Column.MONTHLY_TOTAL_COST)
plt.xticks(
    ticks=range(len(df)),
    labels=df[Column.TIMESTAMP].dt.strftime("%y%m").cast(pl.Int64()).to_list(),
    rotation=90,
    ha="center",
)
# 45 degrees, aligned right


# %% [markdown]
# # Power curve

# %%
df = (
    joined_df.filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
    .group_by(FeatureColumn.YEAR, FeatureColumn.HOUR_OF_DAY)
    .agg(pl.mean(Column.HOURLY_CONSUMPTION))
    .sort(FeatureColumn.HOUR_OF_DAY)
)
sns.lineplot(
    data=df,
    x=FeatureColumn.HOUR_OF_DAY,
    y=Column.HOURLY_CONSUMPTION,
    hue=FeatureColumn.YEAR,
)

# %%
df = (
    joined_df.filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
    .group_by(FeatureColumn.YEAR, FeatureColumn.HOUR_OF_DAY)
    .agg(pl.median(Column.HOURLY_CONSUMPTION))
    .sort(FeatureColumn.HOUR_OF_DAY)
)
sns.lineplot(
    data=df,
    x=FeatureColumn.HOUR_OF_DAY,
    y=Column.HOURLY_CONSUMPTION,
    hue=FeatureColumn.YEAR,
)

# %% [markdown]
# ## By month

# %%
df = (
    joined_df.filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
    .group_by(FeatureColumn.MONTH, FeatureColumn.HOUR_OF_DAY)
    .agg(pl.mean(Column.HOURLY_CONSUMPTION))
    .sort(FeatureColumn.MONTH, FeatureColumn.HOUR_OF_DAY)
    .pivot(
        on=FeatureColumn.HOUR_OF_DAY,
        index=FeatureColumn.MONTH,
        values=Column.HOURLY_CONSUMPTION,
    )
    .to_pandas()
    .set_index(FeatureColumn.MONTH)
)
sns.heatmap(df)

# %% [markdown]
# ## By Event of Day

# %%

# Plot total price timeseries wrt event of day
df = (
    joined_df.filter(pl.col(FeatureColumn.YEAR) >= 2024)
    # .group_by([pl.col(Column.TIMESTAMP).dt.truncate("1mo"), FeatureColumn.EVENT_OF_DAY])
    .group_by([FeatureColumn.MONTH, FeatureColumn.EVENT_OF_DAY])
    .agg(pl.col(Column.HOURLY_TOTAL_COST).sum().alias(Column.MONTHLY_TOTAL_COST))
    .sort(FeatureColumn.MONTH)
    .group_by([FeatureColumn.EVENT_OF_DAY, FeatureColumn.MONTH])
    .agg(pl.col(Column.MONTHLY_TOTAL_COST).rolling_mean(window_size=3, center=True))
)
sns.lineplot(
    data=df,
    x=FeatureColumn.MONTH,
    y=Column.MONTHLY_TOTAL_COST,
    hue=FeatureColumn.EVENT_OF_DAY,
)
plt.xticks(
    # ticks=range(len(df)),
    # labels=df[Column.TIMESTAMP].dt.strftime("%y%m").cast(pl.Int64()).to_list(),
    rotation=45,
    ha="center",
)
# 45 degrees, aligned right


# %% [markdown]
# # Significant Consumption

# %%
df = (
    joined_df.filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
    .group_by(FeatureColumn.YEAR, FeatureColumn.HOUR_OF_DAY)
    .agg(pl.sum(FeatureColumn.SIGNIFICANT_CONSUMPTION))
    .sort(FeatureColumn.HOUR_OF_DAY)
)
sns.lineplot(
    data=df,
    x=FeatureColumn.HOUR_OF_DAY,
    y=FeatureColumn.SIGNIFICANT_CONSUMPTION,
    hue=FeatureColumn.YEAR,
)

# %%
df = (
    joined_df.with_columns(
        pl.when(pl.col(FeatureColumn.WEEKDAY) <= 5)
        .then(pl.lit(1))
        .otherwise(pl.col(FeatureColumn.WEEKDAY))
        .alias(FeatureColumn.WEEKDAY)
    )
    .filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
    .filter(pl.col(FeatureColumn.YEAR) == 2024)
    .group_by(FeatureColumn.WEEKDAY, FeatureColumn.HOUR_OF_DAY)
    .agg(pl.mean(Column.HOURLY_CONSUMPTION))
    .sort(FeatureColumn.HOUR_OF_DAY)
)
sns.lineplot(
    data=df,
    x=FeatureColumn.HOUR_OF_DAY,
    y=Column.HOURLY_CONSUMPTION,
    hue=FeatureColumn.WEEKDAY,
)

# %%
df = (
    joined_df.with_columns(
        pl.when(pl.col(FeatureColumn.WEEKDAY) <= 5)
        .then(pl.lit(1))
        .otherwise(pl.col(FeatureColumn.WEEKDAY))
        .alias(FeatureColumn.WEEKDAY)
    )
    .filter(pl.col(Column.HOURLY_CONSUMPTION) > 0)
    .filter(pl.col(FeatureColumn.YEAR) == 2025)
    .group_by(FeatureColumn.WEEKDAY, FeatureColumn.HOUR_OF_DAY)
    .agg(pl.mean(Column.HOURLY_CONSUMPTION))
    .sort(FeatureColumn.HOUR_OF_DAY)
)
sns.lineplot(
    data=df,
    x=FeatureColumn.HOUR_OF_DAY,
    y=Column.HOURLY_CONSUMPTION,
    hue=FeatureColumn.WEEKDAY,
)

# %% [markdown]
# # What if analysis

# %%
what_if_df = join_prices_and_consumption_data(
    daily_prices_df=prices_df,
    daily_consumption_df=consumption_df.with_columns(
        pl.col(Column.UTC_TIME).dt.offset_by("-2y")
    ),
    monthly_prices_df=None,
)
what_if_df = (
    what_if_df.filter(pl.col(Column.HOURLY_CONSUMPTION).is_not_null())
    .sort(Column.UTC_TIME)
    .with_columns(
        pl.col(Column.HOURLY_CONSUMPTION)
        .rolling_sum_by(Column.UTC_TIME, window_size="1mo")
        .alias(Column.HOURLY_CONSUMPTION_ROLLING_SUM)
    )
)

# %%
df = (
    what_if_df.group_by(
        pl.col(Column.TIMESTAMP).dt.truncate("1mo"), maintain_order=True
    )
    .agg(pl.col(Column.HOURLY_TOTAL_COST).sum().alias(Column.MONTHLY_TOTAL_COST))
    .filter(pl.col(Column.MONTHLY_TOTAL_COST) > 0)
)
sns.barplot(data=df, x=Column.TIMESTAMP, y=Column.MONTHLY_TOTAL_COST)
plt.xticks(
    ticks=range(len(df)),
    labels=df[Column.TIMESTAMP].dt.strftime("%y%m").cast(pl.Int64()).to_list(),
    rotation=90,
    ha="center",
)
# 45 degrees, aligned right


# %%
df = what_if_df.with_columns(
    pl.col(Column.HOURLY_CONSUMPTION_ROLLING_SUM).rolling_mean(window_size=7 * 24)
)
sns.lineplot(data=df, x=Column.TIMESTAMP, y=Column.HOURLY_CONSUMPTION_ROLLING_SUM)
plt.xticks(rotation=45, ha="center")
# 45 degrees, aligned right

# %%


# %%
