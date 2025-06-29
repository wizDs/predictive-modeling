import requests
import datetime
from typing import Sequence
from wiz.budget import schemas

REQUEST_TIMEOUT = datetime.timedelta(seconds=10)
COLUMN_START = "A"
COLUMN_END = "I"


def _get_google_sheet_data(
    *,
    spreadsheet_id: str,
    sheet_name: str,
    api_key: str,
    column_start: str = COLUMN_START,
    column_end: str = COLUMN_END,
) -> Sequence[schemas.Payment]:
    """Gets payment data from Google Sheet and returns sequence of Payment objects."""
    # Construct the URL for the Google Sheets API
    base_url = "https://sheets.googleapis.com/v4/spreadsheets"
    url = (
        f"{base_url}/{spreadsheet_id}/values/{sheet_name}!{column_start}1:{column_end}"
    )

    # Make a GET request to retrieve data from the Google Sheets API
    response = requests.get(
        url=url,
        timeout=REQUEST_TIMEOUT.seconds,
        params={"alt": "json", "key": api_key},
    )
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()


def get_payment_data(
    *,
    spreadsheet_id: str,
    sheet_name: str,
    api_key: str,
    column_start: str = COLUMN_START,
    column_end: str = COLUMN_END,
) -> Sequence[schemas.Payment]:
    """Gets payment data from Google Sheet and returns sequence of Payment objects."""
    data = _get_google_sheet_data(
        spreadsheet_id=spreadsheet_id,
        sheet_name=sheet_name,
        api_key=api_key,
        column_start=column_start,
        column_end=column_end,
    )

    try:
        rows = data["values"]
        _ = rows.pop(0)
        _rows = []
        for row in rows:
            try:
                _rows += [
                    schemas.Payment(**dict(zip(schemas.Payment.model_fields, row)))
                ]
            except Exception as e:
                _row = dict(zip(schemas.Payment.model_fields, row))
                raise RuntimeError(_row) from e

        return _rows

    except requests.exceptions.RequestException as e:
        # Handle any errors that occur during the request
        raise RuntimeError(f"An error occurred: {e}") from e


def get_income_data(
    *,
    spreadsheet_id: str,
    sheet_name: str,
    api_key: str,
    column_start: str = "B",
    column_end: str = "C",
) -> Sequence[schemas.Payment]:
    """Gets payment data from Google Sheet and returns sequence of Payment objects."""
    data = _get_google_sheet_data(
        spreadsheet_id=spreadsheet_id,
        sheet_name=sheet_name,
        api_key=api_key,
        column_start=column_start,
        column_end=column_end,
    )

    try:
        rows = data["values"]
        _ = rows.pop(0)
        _rows = []
        for row in rows:
            try:
                _rows += [schemas.Record(**dict(zip(schemas.Record.model_fields, row)))]
            except Exception as e:
                _row = dict(zip(schemas.Record.model_fields, row))
                raise RuntimeError(_row) from e

        return _rows

    except requests.exceptions.RequestException as e:
        # Handle any errors that occur during the request
        raise RuntimeError(f"An error occurred: {e}") from e
