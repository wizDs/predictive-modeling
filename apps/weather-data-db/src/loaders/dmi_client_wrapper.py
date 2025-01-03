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
        candidate_stations = [
            station.properties.stationId
            for station in self.stations
            if station.properties.name == station_name
        ]

        if not candidate_stations:
            raise ValueError(f"The station '{station_name}' is not valid")

        # Get temperature observations from DMI station in given time period
        for station_id in set(candidate_stations):

            observations = self.client.get_observations(
                parameter=parameter,
                station_id=station_id,
                limit=10_000,
                **kwargs,
            )
            if observations:
                records = map(Record.model_validate, observations)

                # Select only SimpleRecord
                simplerecords = map(operator.attrgetter("simple"), records)

                return list(simplerecords)

        # if entering this loop - then no station_ids can be matched with parameter
        raise ValueError("station_ids can be matched with parameter")

    def get_stations(self) -> list[WeatherStation]:
        stations = map(WeatherStation.model_validate, self.client.get_stations())
        active_stations = filter(lambda s: s.properties.status == "Active", stations)
        return list(active_stations)
