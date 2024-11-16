import requests
import datetime
import polars as pl
from typing import Sequence
import os
from src import payment
from src import schemas

SHEET_ID = os.getenv("SHEET_ID")
API_KEY = os.getenv("API_KEY")
REQUEST_TIMEOUT = datetime.timedelta(seconds=10)
COLUMN_START = "A"
COLUMN_END = "I"


OTHER = 2_500
FOOD = 3_500
BUFFER = 1_000
PROJECT_H = 58_000
PROJECT_S = 15_000

account_saldo = 72_000
monthly_salary = 34_000 + 8_000
additional_variable_cost = FOOD + OTHER + BUFFER
periods = 24

projects = [
    (datetime.date(2024, 12, 1), PROJECT_H),
    (datetime.date(2025, 2, 1), PROJECT_S),
]
projects_df = pl.DataFrame(
    iter(projects),
    schema=pl.Schema({"date": pl.Date, "project_cost": pl.Int64}),
)


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


eval_date = datetime.date.today()
payments = get_google_sheet_data(SHEET_ID, "Fixed & variable costs", API_KEY)
next_date_for_payment = [
    payment.calculate_next_payment(row.starting_point, row.payment_type, eval_date)
    for row in payments
]

payments_df = pl.DataFrame(payments).with_columns(
    next_date_for_payment=pl.Series(next_date_for_payment)
)
df = (
    pl.DataFrame(
        payment.calculate_total_payments(
            eval_date=eval_date, payments=payments, monthly_periods=periods
        ),
        schema=pl.Schema({"date": pl.Date, "living_cost": pl.Float32}),
    )
    .join(other=projects_df, on=["date"], how="left")
    .with_columns(project_cost=pl.col("project_cost").fill_null(0))
    .with_columns(salary=pl.Series([monthly_salary] * periods))
    .with_columns(additional_cost=pl.Series([additional_variable_cost] * periods))
    .with_columns(
        delta=(
            pl.col("salary")
            - pl.col("project_cost")
            - pl.col("living_cost")
            - pl.col("additional_cost")
        )
    )
    .with_columns(saldo=account_saldo + pl.col("delta").cum_sum())
)
print(df.to_pandas())
print(df.to_pandas())
