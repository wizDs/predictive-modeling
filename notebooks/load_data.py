import enum
from typing import Self
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from collections.abc import Iterable, Sequence
import datetime
import pandas as pd
import requests
import pydantic
import polars as pl

_TIMEOUT = datetime.timedelta(seconds=30)
_LIMIT = 1000
"""Maximum number of records per request"""
V1_DEPRECATED = datetime.date(2025, 10, 1)
MAX_WORKERS = 4


class Frequency(enum.IntEnum):
    HOURLY = 24
    QUARTER_HOURLY = 4 * 24


class Elspotprices(pydantic.BaseModel):
    HourUTC: pydantic.PastDatetime | pydantic.PastDate
    HourDK: pydantic.PastDatetime | pydantic.PastDate | None = None
    # PriceArea: str | None = None
    SpotPriceDKK: float | None = None
    # SpotPriceEUR: float | None = None


class BaseRequestParams(pydantic.BaseModel):
    start: pydantic.PastDate
    end: pydantic.PastDate
    limit: pydantic.NonNegativeInt = _LIMIT
    offset: pydantic.NonNegativeInt
    sort: str | None = None
    order: str = "asc"
    filter: str

    @pydantic.model_validator(mode="after")
    def validate_dates(self) -> Self:
        if self.start > self.end:
            raise ValueError("Start date must be before end date")
        return self


class RequestParamsV1(BaseRequestParams):
    sort: str | None = "HourUTC"
    filter: str = '{"PriceArea": "DK1"}'


class RequestParamsV2(BaseRequestParams):
    sort: str | None = "TimeUTC"
    filter: str = '{"PriceArea": ["DK1"]}'


RequestParams = RequestParamsV1 | RequestParamsV2


class EnergyDataClient:
    def __init__(self):
        self._URL_V1 = "https://api.energidataservice.dk/dataset/Elspotprices"
        self._URL_V2 = "https://api.energidataservice.dk/dataset/DayAheadPrices"

    def _get_url(self, params: RequestParams) -> str:
        match params:
            case RequestParamsV1():
                return self._URL_V1
            case RequestParamsV2():
                return self._URL_V2
            case _:
                raise ValueError(f"Invalid params: {params}")

    def get_single_response(self, parameters: RequestParams) -> Sequence[Elspotprices]:
        response: requests.Response = requests.get(
            self._get_url(parameters),
            params=parameters.model_dump(exclude_none=True),
            timeout=_TIMEOUT.seconds,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.status_code}")

        records = response.json()["records"]
        match parameters:
            case RequestParamsV1():
                return [Elspotprices.model_validate(item) for item in records]
            case RequestParamsV2():
                return [
                    Elspotprices.model_validate(
                        {
                            "HourUTC": item["TimeUTC"],
                            "HourDK": item["TimeDK"],
                            "PriceArea": item["PriceArea"],
                            "SpotPriceDKK": item["DayAheadPriceDKK"],
                        }
                    )
                    for item in records
                ]
            case _:
                raise ValueError(f"Invalid params: {parameters}")

    def get(
        self, seq_parameters: Iterable[RequestParams], n_jobs: int = 1
    ) -> Sequence[Elspotprices]:
        stacked_prices: list[Sequence[Elspotprices]]
        if n_jobs > 1:
            stacked_prices = joblib.Parallel(n_jobs=MAX_WORKERS)(
                joblib.delayed(client.get_single_response)(parameters=params)
                for params in seq_parameters
            )
        elif n_jobs == 1:
            stacked_prices = []
            for params in seq_parameters:
                stacked_prices.append(self.get_single_response(params))
        else:
            raise ValueError(f"Invalid number of jobs: {n_jobs}")

        # Flatten the requests into a single list of prices
        return [price for records in stacked_prices for price in records]


def get_expected_record_count(
    start: datetime.date, end: datetime.date, frequency: Frequency
) -> int:
    return (end - start).days * frequency.value


start = datetime.date(2020, 1, 1)
end = datetime.date(2025, 12, 5)


def split_request_params(
    start: datetime.date,
    end: datetime.date,
    request_schema: type[RequestParams],
) -> Iterable[RequestParams]:
    if issubclass(request_schema, RequestParamsV1):
        frequency = Frequency.HOURLY
    elif issubclass(request_schema, RequestParamsV2):
        frequency = Frequency.QUARTER_HOURLY
    else:
        raise ValueError(f"Invalid request schema: {request_schema}")

    expected_records = get_expected_record_count(start, end, frequency)

    if expected_records <= _LIMIT:
        yield request_schema(start=start, end=end, limit=_LIMIT, offset=0)
    else:
        days_size = _LIMIT // frequency.value  # 24 hours per day

        local_start = start
        local_end = start + datetime.timedelta(days=days_size)

        while local_end < end:
            yield request_schema(
                start=local_start, end=local_end, limit=_LIMIT, offset=0
            )
            local_start = local_end
            local_end = local_start + datetime.timedelta(days=days_size)

        # Last request
        yield request_schema(start=local_start, end=end, limit=_LIMIT, offset=0)


def split_request_params_all_versions(
    start: datetime.date, end: datetime.date
) -> Iterable[RequestParamsV1]:
    match start, end:
        case (start, end) if start < V1_DEPRECATED and end < V1_DEPRECATED:
            for params in split_request_params(start, end, RequestParamsV1):
                yield params
        case (start, end) if start < V1_DEPRECATED and end >= V1_DEPRECATED:
            for params in split_request_params(start, V1_DEPRECATED, RequestParamsV1):
                yield params
            for params in split_request_params(V1_DEPRECATED, end, RequestParamsV2):
                yield params
        case (start, end) if start >= V1_DEPRECATED and end >= V1_DEPRECATED:
            for params in split_request_params(start, end, RequestParamsV2):
                yield params
        case _:
            raise ValueError(f"Invalid date range: {start} - {end}")


seq_params = split_request_params_all_versions(start, end)
client = EnergyDataClient()
prices: Sequence[Elspotprices] = client.get(
    seq_parameters=seq_params, n_jobs=MAX_WORKERS
)

prices_df: pl.DataFrame = (
    pl.DataFrame(prices)
    .group_by(pl.col("HourUTC").dt.truncate("1h"), maintain_order=True)
    .agg(pl.col("SpotPriceDKK").mean().alias("SpotPriceDKK"))
    .with_columns(price_kwh_in_dkk=pl.col("SpotPriceDKK") / 1000)
)


class PricesMonthly(pydantic.BaseModel):
    year: pydantic.NonNegativeInt = pydantic.Field(ge=2010, le=2100)
    month: pydantic.NonNegativeInt = pydantic.Field(ge=1, le=12)
    monthly_price_kwh_in_dkk: float


prices_monthly_df: pl.DataFrame = pl.DataFrame(
    [
        PricesMonthly(
            year=2023,
            month=6,
            monthly_price_kwh_in_dkk=96.52,
        ),
        PricesMonthly(
            year=2023,
            month=7,
            monthly_price_kwh_in_dkk=110.55,
        ),
        PricesMonthly(
            year=2023,
            month=8,
            monthly_price_kwh_in_dkk=100.83,
        ),
        PricesMonthly(
            year=2023,
            month=9,
            monthly_price_kwh_in_dkk=108.62,
        ),
        PricesMonthly(
            year=2023,
            month=10,
            monthly_price_kwh_in_dkk=98.80,
        ),
        PricesMonthly(
            year=2023,
            month=11,
            monthly_price_kwh_in_dkk=112.17,
        ),
        PricesMonthly(
            year=2023,
            month=12,
            monthly_price_kwh_in_dkk=108.81,
        ),
        PricesMonthly(
            year=2024,
            month=1,
            monthly_price_kwh_in_dkk=105.91,
        ),
        PricesMonthly(
            year=2024,
            month=2,
            monthly_price_kwh_in_dkk=91.65,
        ),
        PricesMonthly(
            year=2024,
            month=3,
            monthly_price_kwh_in_dkk=76.01,
        ),
        PricesMonthly(
            year=2024,
            month=4,
            monthly_price_kwh_in_dkk=67.39,
        ),
        PricesMonthly(
            year=2024,
            month=5,
            monthly_price_kwh_in_dkk=70.84,
        ),
        PricesMonthly(
            year=2024,
            month=6,
            monthly_price_kwh_in_dkk=78.67,
        ),
        PricesMonthly(
            year=2024,
            month=7,
            monthly_price_kwh_in_dkk=82.04,
        ),
        PricesMonthly(
            year=2024,
            month=8,
            monthly_price_kwh_in_dkk=80.63,
        ),
        PricesMonthly(
            year=2024,
            month=9,
            monthly_price_kwh_in_dkk=94.69,
        ),
        PricesMonthly(
            year=2024,
            month=10,
            monthly_price_kwh_in_dkk=81.68,
        ),
        PricesMonthly(
            year=2024,
            month=11,
            monthly_price_kwh_in_dkk=98.99,
        ),
        PricesMonthly(
            year=2024,
            month=12,
            monthly_price_kwh_in_dkk=111.03,
        ),
        PricesMonthly(
            year=2025,
            month=1,
            monthly_price_kwh_in_dkk=111.91,
        ),
        PricesMonthly(
            year=2025,
            month=2,
            monthly_price_kwh_in_dkk=92.95,
        ),
        PricesMonthly(
            year=2025,
            month=3,
            monthly_price_kwh_in_dkk=99.29,
        ),
        PricesMonthly(
            year=2025,
            month=4,
            monthly_price_kwh_in_dkk=86.11,
        ),
        PricesMonthly(
            year=2025,
            month=5,
            monthly_price_kwh_in_dkk=72.94,
        ),
        PricesMonthly(
            year=2025,
            month=6,
            monthly_price_kwh_in_dkk=85.06,
        ),
        PricesMonthly(
            year=2025,
            month=7,
            monthly_price_kwh_in_dkk=87.68,
        ),
        PricesMonthly(
            year=2025,
            month=8,
            monthly_price_kwh_in_dkk=88.87,
        ),
        PricesMonthly(
            year=2025,
            month=9,
            monthly_price_kwh_in_dkk=90.54,
        ),
        PricesMonthly(
            year=2025,
            month=10,
            monthly_price_kwh_in_dkk=97.11,
        ),
        PricesMonthly(
            year=2025,
            month=11,
            monthly_price_kwh_in_dkk=114.31,
        ),
        PricesMonthly(
            year=2025,
            month=12,
            monthly_price_kwh_in_dkk=116.08,
        ),
    ]
).with_columns(pl.col("monthly_price_kwh_in_dkk") / 100)

prices_monthly_ts = (
    pl.DataFrame()
    .with_columns(pl.date_range(start=start, end=end, interval="1d").alias("timestamp"))
    .with_columns(
        year=pl.col("timestamp").dt.year(),
        month=pl.col("timestamp").dt.month(),
    )
    .join(other=prices_monthly_df, on=["year", "month"], how="inner")
)

# Load consumption data
consumption_df: pl.DataFrame = pl.read_csv(
    source="/Users/wiz/projects/predictive-modeling/data/energi-data.csv",
    decimal_comma=True,
    schema={
        "HourUTC": pl.Datetime,
        "SpotPriceDKK": pl.Float64,
    },
).rename({"SpotPriceDKK": "consumption_kwh_hourly"})

# Join prices and consumption data
joined_df = (
    prices_df.drop("SpotPriceDKK")
    .join(other=consumption_df, on="HourUTC", how="left")
    .with_columns(
        pl.col("HourUTC").dt.convert_time_zone("Europe/Copenhagen").alias("timestamp")
    )
    # .filter(pl.col("timestamp").dt.year() >= 2024)
    .with_columns(pl.col("timestamp").dt.hour().alias("hour_of_day"))
    .with_columns(pl.col("timestamp").dt.day().alias("day_of_month"))
    .with_columns(pl.col("timestamp").dt.month().alias("month"))
    .with_columns(pl.col("timestamp").dt.year().alias("year"))
    .with_columns(pl.col("timestamp").dt.weekday().alias("weekday"))
    .with_columns(
        event_of_day=pl.when(pl.col("hour_of_day").is_between(6, 10, closed="left"))
        .then(pl.lit("Morning"))  # 0
        .when(pl.col("hour_of_day").is_between(10, 18, closed="left"))
        .then(pl.lit("Afternoon"))  # 1
        .when(pl.col("hour_of_day").is_between(18, 22, closed="left"))
        .then(pl.lit("Evening"))  # 2
        .when(
            pl.col("hour_of_day").is_between(22, 24, closed="both")
            | pl.col("hour_of_day").is_between(0, 6, closed="left")
        )
        .then(pl.lit("Night"))  # 3
        .otherwise(-99)
    )
    .join(other=prices_monthly_df, on=["year", "month"], how="left")
    .with_columns(
        consumption_in_dkk=pl.col("price_kwh_in_dkk") * pl.col("consumption_kwh_hourly")
    )
    .with_columns(
        consumption_in_dkk_status_quo=pl.col("monthly_price_kwh_in_dkk")
        * pl.col("consumption_kwh_hourly")
    )
)

# Mean price by event of day
joined_df.group_by("year", "event_of_day").agg(
    pl.col("price_kwh_in_dkk").mean().alias("price_kwh_in_dkk_mean")
).sort(by=["year", "event_of_day"])

# Mean kwh consumption by event of day
joined_df.group_by("year", "event_of_day").agg(
    pl.col("consumption_kwh_hourly").sum().alias("consumption_kwh_hourly")
).sort(by=["year", "event_of_day"])


# Mean dkk spent by event of day
joined_df.group_by("year", "event_of_day").agg(
    pl.col("consumption_in_dkk").sum().alias("consumption_in_dkk_sum")
).sort(by=["year", "event_of_day"])


sns.set_style("whitegrid")

# Plot prices timeseries
df_pd = (
    joined_df.group_by(pl.col("timestamp").dt.truncate("1mo"))
    .agg(
        [
            pl.col("price_kwh_in_dkk").mean().alias("price_kwh_in_dkk_mean"),
            pl.col("price_kwh_in_dkk").std().alias("price_kwh_in_dkk_std"),
            pl.col("price_kwh_in_dkk").max().alias("max"),
            pl.col("price_kwh_in_dkk").min().alias("min"),
        ]
    )
    .with_columns(
        [
            (pl.col("price_kwh_in_dkk_mean") + pl.col("price_kwh_in_dkk_std")).alias(
                "upper"
            ),
            (pl.col("price_kwh_in_dkk_mean") - pl.col("price_kwh_in_dkk_std")).alias(
                "lower"
            ),
        ]
    )
    .to_pandas()
)
sns.lineplot(data=df_pd, x="timestamp", y="price_kwh_in_dkk_mean")
sns.lineplot(data=df_pd, x="timestamp", y="upper", color="green")
sns.lineplot(data=df_pd, x="timestamp", y="lower", color="green")
sns.lineplot(data=df_pd, x="timestamp", y="max", color="red")
sns.lineplot(data=df_pd, x="timestamp", y="min", color="red")
sns.lineplot(
    data=prices_monthly_ts, x="timestamp", y="monthly_price_kwh_in_dkk", color="grey"
)
plt.show()


# Plot consumption timeseries
consumption_df_pd = (
    joined_df.filter(pl.col("year") >= 2024)
    .group_by([pl.col("timestamp").dt.truncate("1mo"), "event_of_day"])
    .agg(pl.col("consumption_kwh_hourly").sum().alias("consumption_kwh_sum"))
    .to_pandas()
)
sns.lineplot(
    data=consumption_df_pd, x="timestamp", y="consumption_kwh_sum", hue="event_of_day"
)
# sns.lineplot(data=consumption_df_pd, x="HourDK", y="upper", color="red")
# sns.lineplot(data=consumption_df_pd, x="HourDK", y="lower", color="green")
plt.show()


# Plot total price timeseries wrt event of day
consumption_df_pd = (
    joined_df.filter(pl.col("year") >= 2024)
    .group_by([pl.col("timestamp").dt.truncate("1mo"), "event_of_day"])
    .agg(pl.col("price_kwh_in_dkk").sum().alias("price_kwh_in_dkk_sum"))
    .to_pandas()
)
sns.lineplot(
    data=consumption_df_pd, x="timestamp", y="price_kwh_in_dkk_sum", hue="event_of_day"
)
# sns.lineplot(data=consumption_df_pd, x="HourDK", y="upper", color="red")
# sns.lineplot(data=consumption_df_pd, x="HourDK", y="lower", color="green")
plt.show()


# Plot total price timeseries vs baseline
consumption_df_pd = (
    joined_df.filter(pl.col("year") >= 2024)
    .group_by([pl.col("timestamp").dt.truncate("1mo")], maintain_order=True)
    .agg(
        [
            pl.col("consumption_in_dkk_status_quo")
            .sum()
            .alias("consumption_in_dkk_status_quo_sum"),
            pl.col("consumption_in_dkk").sum().alias("consumption_in_dkk_sum"),
        ]
    )
    .with_columns(
        difference=pl.col("consumption_in_dkk_sum")
        - pl.col("consumption_in_dkk_status_quo_sum")
    )
    .with_columns(
        difference_percentage=pl.col("difference")
        / pl.col("consumption_in_dkk_status_quo_sum")
        * 100
    )
    .to_pandas()
)
sns.lineplot(data=consumption_df_pd, x="timestamp", y="consumption_in_dkk_sum")
sns.lineplot(
    data=consumption_df_pd, x="timestamp", y="consumption_in_dkk_status_quo_sum"
)
# sns.lineplot(data=consumption_df_pd, x="HourDK", y="upper", color="red")
# sns.lineplot(data=consumption_df_pd, x="HourDK", y="lower", color="green")
plt.show()


# Plot consumption annually
consumption_df_pd = (
    consumption_df.group_by(pl.col("timestamp").dt.truncate("1y"))
    .agg(pl.col("consumption_kwh_hourly").sum().alias("consumption_kwh_sum"))
    .to_pandas()
)
sns.lineplot(data=consumption_df_pd, x="timestamp", y="consumption_kwh_sum")
# sns.lineplot(data=consumption_df_pd, x="HourDK", y="upper", color="red")
# sns.lineplot(data=consumption_df_pd, x="HourDK", y="lower", color="green")
plt.show()

print(1)
