import os
import argparse
import logging
from datetime import datetime
import dmi_open_data as dmi  # type: ignore[import-untyped]
import dotenv

from src import pipelines
from src import interfaces


def get_apikey() -> str:
    dotenv.load_dotenv()
    return os.environ["APIKEY_METOPS"]


def get_args() -> interfaces.WeatherInterface:

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
        choices=["spotprice", "powersystem", "weather"],
    )
    parser.add_argument(
        "--station",
        default="Københavns Lufthavn",
        type=str,
        help="the station name where the observations are collected",
    )

    args = parser.parse_args()

    return interfaces.WeatherInterface(
        start_date=args.start_date,
        end_date=args.end_date,
        freq=args.freq,
        path=args.path,
        apikey=get_apikey(),
        station_name=args.station,
        datatype=dmi.Parameter(args.datatype),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_args()
    try:
        pipelines.WeatherPipeline(config).run()
    except Warning as w:
        logging.warning(w)
