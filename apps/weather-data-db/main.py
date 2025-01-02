import os
import argparse
import logging
import dmi_open_data as dmi  # type: ignore[import-untyped]
import dotenv

from src import pipelines
from src import interfaces


def get_apikey() -> str:
    dotenv.load_dotenv()
    return os.environ["APIKEY_METOPS"]


def get_interface_from_args(args: argparse.Namespace) -> interfaces.WeatherInterface:
    return interfaces.WeatherInterface(
        start_date=args.start_date,
        end_date=args.end_date,
        freq=args.freq,
        path=args.path,
        apikey=get_apikey(),
        station_name=args.station,
        datatype=dmi.Parameter(args.datatype),
        time_partition=args.time_partition,
    )


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="run datapipeline for 1 month")
    parser.add_argument(
        "--start_date",
        type=str,
        help="start date for which the data pipeline should be run",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="end date for which the data pipeline should be run",
    )
    parser.add_argument("--freq", default="MS", type=str, help="partitioning frequency")
    parser.add_argument(
        "--path",
        default="./data",
        type=str,
        help="the destination where data will be persisted",
    )
    parser.add_argument(
        "--datatype",
        default="temp_dry",
        type=str,
        help="type of datapipeline to be run",
    )
    parser.add_argument(
        "--station",
        default="Københavns Lufthavn",
        type=str,
        help="the station name where the observations are collected",
    )
    parser.add_argument(
        "--time_partition",
        default="MONTH",
        type=str,
        help="persist the observations are collected with this time partition",
    )
    parser.add_argument(
        "--pipeline_type",
        default="dmi",
        type=str,
        help=(
            "determine if you want to acquire data from dmi "
            "or make some transformations to acquired dmi data"
        ),
        choices=["dmi", "transformation"],
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arguments = get_args()
    interface = get_interface_from_args(args=arguments)
    try:
        match arguments.pipeline_type:
            case "dmi":
                pipelines.WeatherPipeline(interface).run()
            case "transformation":
                pipelines.WeatherDailyAggregationPipeline(interface).run()

    except Warning as w:
        logging.warning(w)
