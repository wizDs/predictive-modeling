import operator
import dmi_open_data as dmi  # type: ignore[import-untyped]
from src.models.weatherstation import WeatherStation  # type: ignore[import-untyped]
from src.models.record import Record, SimpleRecord  # type: ignore[import-untyped]


class DMIClientWrapper:

    def __init__(self, api_key: str):
        self.client = dmi.DMIOpenDataClient(api_key=api_key)
        self.stations = self.get_stations()

    def get(
        self, station_name: str, parameter: dmi.Parameter, **kwargs
    ) -> list[SimpleRecord]:

        # Identify station from station_name
        try:
            station = next(
                station
                for station in self.stations
                if station.properties.name == station_name
            )
        except StopIteration as exc:
            raise ValueError(f"The station '{station_name}' is not valid") from exc

        if parameter.value not in set(station.properties.parameterId):
            return []

        # Get temperature observations from DMI station in given time period
        observations = self.client.get_observations(
            parameter=parameter,
            station_id=station.properties.stationId,
            limit=10_000,
            **kwargs,
        )
        records = map(Record.model_validate, observations)

        # Select only SimpleRecord
        simplerecords = map(operator.attrgetter("simple"), records)

        return list(simplerecords)

    def get_stations(self) -> list[WeatherStation]:
        stations = map(WeatherStation.model_validate, self.client.get_stations())
        active_stations = filter(lambda s: s.properties.status == "Active", stations)
        return list(active_stations)
