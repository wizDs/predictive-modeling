import datetime
import logging
import pathlib
from typing import Iterable, assert_never
import uuid
import itertools
import pandas as pd
from src import interfaces  # type: ignore[import-untyped]
from src.loaders.dmi_client_wrapper import DMIClientWrapper  # type: ignore[import-untyped]


def full_path_from_interface(
    interface: interfaces.WeatherInterface,
) -> Iterable[pathlib.Path]:
    base_path = interface.path / interface.datatype.value
    return iter([base_path])
    # for d in interface.date_range:
    #     match interface.freq:
    #         case interfaces.TimeDelta.DAY:
    #             yield base_path / f"/year={d.year}" / "month={d.month}/" / "day={d.day}"
    #         case interfaces.TimeDelta.MONTH:
    #             yield base_path
    #         case interfaces.TimeDelta.YEAR:
    #             raise ValueError(f"{interfaces.TimeDelta.YEAR} not supported")
    #         case interfaces.TimeDelta.WEEK:
    #             raise ValueError(f"{interfaces.TimeDelta.WEEK} not supported")
    #         case interfaces.TimeDelta.QUARTER:
    #             raise ValueError(f"{interfaces.TimeDelta.QUARTER} not supported")
    #         case _:
    #             assert_never(interface.freq)


class DataPipeline:

    def __init__(self, interface: interfaces.WeatherInterface) -> None:
        self.interface = interface

    def persist_to_parquet(self, data: pd.DataFrame) -> None:

        path = next(full_path_from_interface(self.interface))
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        data.to_parquet(path=path / f"{uuid.uuid4()}.parquet")

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

        # persist data
        logging.info(
            "%s-data for %s-%s stored to %s (%s rows)",
            self.interface.datatype.value,
            start_date.date(),
            end_date.date(),
            self.interface.path,
            len(data),
        )
        self.persist_to_parquet(data=data)

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
            raise Warning(f"no data was found for {start.date()}-{end.date()}")

        # Create pandas data frame
        data = pd.DataFrame([row.model_dump() for row in simplerecords])
        data["station"] = self.interface.station_name
        return data

    def _extract_partitioned(
        self, client: DMIClientWrapper, date_range: list[datetime.datetime]
    ) -> Iterable[pd.DataFrame]:
        for idx, curr_date in enumerate(date_range):
            if idx != 0:
                logging.info("%s-%s", prev_date.date(), curr_date.date())
                yield self._extract_simple(
                    client=client, start=prev_date, end=curr_date
                )
            prev_date = curr_date
