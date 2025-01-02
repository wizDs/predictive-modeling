import pathlib
from typing import Iterable, assert_never
import datetime
import dmi_open_data as dmi  # type: ignore[import-untyped]
import pydantic
import enum
import pandas as pd


class TimeDelta(enum.StrEnum):
    DAY = "D"
    WEEK = "W"
    MONTH = "MS"
    QUARTER = "QS"
    YEAR = "YS"


class WeatherInterface(pydantic.BaseModel):
    start_date: pydantic.PastDatetime
    end_date: pydantic.PastDatetime
    freq: TimeDelta = TimeDelta.MONTH
    path: pathlib.Path
    apikey: str
    station_name: str
    datatype: dmi.Parameter

    @property
    def date_range(self) -> list[datetime.date]:
        _date_range = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.freq
        ).tolist()
        _all_date_range = [d.to_pydatetime() for d in _date_range] + [
            self.start_date,
            self.end_date,
        ]
        return sorted(set(_all_date_range))
