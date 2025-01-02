import abc
import datetime
import itertools
import logging
import pathlib
import uuid
from typing import Iterable, assert_never, final

import delta
import pandas as pd
import pyspark
import tqdm  # type: ignore[import-untyped]
from pyspark.sql import functions as f
from src.loaders.dmi_client_wrapper import DMIClientWrapper

from src import interfaces  # type: ignore[import-untyped]

DELTA_RETENTION_HOURS = 1


def delta_spark_session_builder() -> pyspark.sql.SparkSession:
    builder = (
        pyspark.sql.SparkSession.builder.appName("MyApp")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = delta.configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def path_from_interface(
    interface: interfaces.WeatherInterface,
) -> pathlib.Path:
    base_path = interface.path / interface.datatype.value
    return base_path


class DataPipeline(abc.ABC):

    def __init__(self, interface: interfaces.WeatherInterface) -> None:
        self.interface = interface

    def run(self): ...


class WeatherPipeline(DataPipeline):

    def run(self):

        start_date = self.interface.start_date
        end_date = self.interface.end_date

        dmi_client = DMIClientWrapper(self.interface.apikey)
        if (len(self.interface.date_range) == 2) and (
            (end_date - start_date).days < 35
        ):
            data = self._extract_simple(
                client=dmi_client, start=start_date, end=end_date
            )
        else:
            partitioned_data = self._extract_partitioned(
                dmi_client, self.interface.date_range
            )
            data = pd.concat(partitioned_data, ignore_index=True)

        # persist stations
        dmi_stations = pd.DataFrame(s.model_dump() for s in dmi_client.stations)
        dmi_stations.to_parquet(self.interface.path / "dmi_stations.parquet")

        # persist data
        logging.info(
            "%s-data for %s-%s acquired (%s rows)",
            self.interface.datatype.value,
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
            len(data),
        )
        self.persist_to_delta(
            data=data,
            table_path=self.interface.path / "weather_data",
            time_partition=self.interface.time_partition,
        )

    def _extract_simple(
        self,
        client: DMIClientWrapper,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> pd.DataFrame:

        # Extract data from energidataservice
        simplerecords = client.get(
            station_name=self.interface.station_name,
            parameter=self.interface.datatype,
            from_time=start,
            to_time=end,
        )
        if not simplerecords:
            raise RuntimeError(
                f"no data was found for {start.strftime("%Y%m%d")}-{end.strftime("%Y%m%d")}"
            )

        # Create pandas data frame
        data = pd.DataFrame([row.model_dump() for row in simplerecords])
        data["data_type"] = self.interface.datatype.value
        data["station"] = self.interface.station_name
        data["month_key"] = data["time"].dt.year * 100 + data["time"].dt.month
        data["day_key"] = (
            data["time"].dt.year * 10000
            + data["time"].dt.month * 100
            + data["time"].dt.day
        )
        data["hour"] = data["time"].dt.hour
        return data

    def _extract_partitioned(
        self, client: DMIClientWrapper, date_range: list[datetime.datetime]
    ) -> Iterable[pd.DataFrame]:
        iteration_count = len(date_range) - 1
        for idx, curr_date in tqdm.tqdm(enumerate(date_range), total=iteration_count):
            if idx != 0:
                try:
                    yield self._extract_simple(
                        client=client, start=prev_date, end=curr_date
                    )
                except Exception as e:
                    raise ValueError(
                        f"A problem occured in {prev_date.date()}-{curr_date.date()}"
                    ) from e
            prev_date = curr_date

    def persist_to_delta(
        self,
        data: pd.DataFrame,
        table_path: pathlib.Path,
        time_partition: interfaces.TimeDelta = interfaces.TimeDelta.MONTH,
    ) -> None:

        spark = delta_spark_session_builder()

        df = spark.createDataFrame(data)
        match time_partition:
            case interfaces.TimeDelta.DAY:
                time_key = tuple(data["day_key"].unique().tolist())
                replace_time = (
                    f"day_key in {time_key}"
                    if len(time_key) > 1
                    else f"day_key = {time_key[0]}"
                )
                partition_columns = [
                    "station",
                    "data_type",
                    "month_key",
                    "day_key",
                ]
            case interfaces.TimeDelta.MONTH:
                time_key = tuple(data["month_key"].unique().tolist())
                replace_time = (
                    f"month_key in {time_key}"
                    if len(time_key) > 1
                    else f"month_key = {time_key[0]}"
                )
                partition_columns = [
                    "station",
                    "data_type",
                    "month_key",
                ]
            case _:
                raise ValueError("Only `Month` is supported at the moment")

        replace_where = (
            f"station = '{self.interface.station_name}' AND "
            f"data_type = '{self.interface.datatype.value}' AND "
        )
        df.write.format("delta").mode("overwrite").partitionBy(
            partition_columns
        ).option("replaceWhere", replace_where + replace_time).save(
            table_path.as_posix()
        )
        logging.info(
            "%s-data for %s-%s stored to %s (%s rows)",
            self.interface.datatype.value,
            self.interface.start_date.strftime("%Y%m%d"),
            self.interface.end_date.strftime("%Y%m%d"),
            table_path.as_posix(),
            len(data),
        )
        delta_table = delta.DeltaTable.forPath(spark, table_path.as_posix())
        delta_table.vacuum(DELTA_RETENTION_HOURS)


class WeatherDailyAggregationPipeline(DataPipeline):

    def run(self):

        start_date = self.interface.start_date
        end_date = self.interface.end_date
        input_table_name = self.interface.path / "weather_data"
        output_table_name = self.interface.path / "daily_weather_data"

        spark = delta_spark_session_builder()
        data = (
            spark.read.format("delta")
            .load(input_table_name.as_posix())
            .where(
                f.col("day_key").between(
                    start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
                )
            )
            .where(f.col("data_type") == self.interface.datatype.value)
            .where(f.col("station") == self.interface.station_name)
        )
        agg_data = (
            data.groupby(["day_key", "month_key", "data_type", "station"])
            .agg(f.avg("value").alias("avg_value"))
            .orderBy(["day_key", "month_key", "data_type", "station"])
        )

        # persist data
        logging.info(
            "%s-data for %s-%s loaded from %s and then aggregated (%s rows)",
            self.interface.datatype.value,
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
            input_table_name.as_posix(),
            agg_data.count(),
        )
        self.persist_to_delta(
            data=agg_data,
            table_path=output_table_name,
            time_partition=self.interface.time_partition,
            spark=spark,
        )

    def persist_to_delta(
        self,
        data: pd.DataFrame | pyspark.sql.DataFrame,
        table_path: pathlib.Path,
        time_partition: interfaces.TimeDelta = interfaces.TimeDelta.MONTH,
        spark: pyspark.sql.SparkSession | None = None,
    ) -> None:
        if spark is None:
            spark = delta_spark_session_builder()

        match data:
            case pd.DataFrame():
                df = spark.createDataFrame(data)
            case pyspark.sql.DataFrame():
                df = data
                data = df.toPandas()
            case _:
                assert_never(data)

        match time_partition:
            case interfaces.TimeDelta.DAY:
                time_key = tuple(data["day_key"].unique().tolist())
                replace_time = (
                    f"day_key in {time_key}"
                    if len(time_key) > 1
                    else f"day_key = {time_key[0]}"
                )
                partition_columns = [
                    "station",
                    "data_type",
                    "month_key",
                    "day_key",
                ]
            case interfaces.TimeDelta.MONTH:
                time_key = tuple(data["month_key"].unique().tolist())
                replace_time = (
                    f"month_key in {time_key}"
                    if len(time_key) > 1
                    else f"month_key = {time_key[0]}"
                )
                partition_columns = [
                    "station",
                    "data_type",
                    "month_key",
                ]
            case _:
                raise ValueError("Only `Month` and `DAY` are supported at the moment")

        replace_where = (
            f"station = '{self.interface.station_name}' AND "
            f"data_type = '{self.interface.datatype.value}' AND "
        )
        df.write.format("delta").mode("overwrite").partitionBy(
            partition_columns
        ).option("replaceWhere", replace_where + replace_time).save(
            table_path.as_posix()
        )
        logging.info(
            "%s-data for %s-%s stored to %s (%s rows)",
            self.interface.datatype.value,
            self.interface.start_date.strftime("%Y%m%d"),
            self.interface.end_date.strftime("%Y%m%d"),
            table_path.as_posix(),
            len(data),
        )
        delta_table = delta.DeltaTable.forPath(spark, table_path.as_posix())
        delta_table.vacuum(DELTA_RETENTION_HOURS)
