import requests
import datetime
from typing import Sequence
from src import schemas

REQUEST_TIMEOUT = datetime.timedelta(seconds=10)
COLUMN_START = "A"
COLUMN_END = "I"


def get_google_sheet_data(
    spreadsheet_id: str, sheet_name: str, api_key: str
) -> Sequence[schemas.Payment]:
    """"""
    # Construct the URL for the Google Sheets API
    base_url = "https://sheets.googleapis.com/v4/spreadsheets"
    url = (
        f"{base_url}/{spreadsheet_id}/values/{sheet_name}!{COLUMN_START}1:{COLUMN_END}"
    )

    try:
        # Make a GET request to retrieve data from the Google Sheets API
        response = requests.get(
            url=url,
            timeout=REQUEST_TIMEOUT.seconds,
            params={"alt": "json", "key": api_key},
        )
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        data = response.json()
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
