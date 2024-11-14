import pandas as pd
import requests
import datetime
import polars as pl
import pydantic
from typing import Any, Literal, Optional, Sequence, assert_never
import enum
from dateutil import relativedelta
import os

SHEET_ID = os.getenv("SHEET_ID")
API_KEY = os.getenv("API_KEY")
REQUEST_TIMEOUT = datetime.timedelta(seconds=10)
COLUMN_START = "A"
COLUMN_END = "I"


class PaymentType(enum.StrEnum):
    Monthly = "Måned"
    Annually = "År"
    Quarterly = "Kvartal"
    BiAnnually = "Halvårligt"


class Payment(pydantic.BaseModel):
    cost_type: Literal["Fixed", "Variabel"]
    category: str
    sub_category: str
    description: str
    payment_type: PaymentType
    price: pydantic.NonNegativeFloat
    annual_price: Optional[pydantic.NonNegativeFloat] = None
    monthly_price: Optional[pydantic.NonNegativeFloat] = None
    starting_point: Optional[str] = None

    @pydantic.field_validator("price", mode="before")
    @classmethod
    def transform_price(cls, x: Optional[str | float]) -> Optional[str | float]:
        return cls._replace_comma(x)

    @pydantic.field_validator("annual_price", mode="before")
    @classmethod
    def transform_annual_price(cls, x: Optional[str]) -> Optional[str]:
        return cls._replace_comma(x)

    @pydantic.field_validator("monthly_price", mode="before")
    @classmethod
    def transform_monthly_price(cls, x: Optional[str]) -> Optional[str]:
        return cls._replace_comma(x)

    @staticmethod
    def _replace_comma(x: Any) -> Any:
        if isinstance(x, str):
            if "," in x:
                x = x.replace(",", ".")
        return x

    @staticmethod
    def calculate_annual_price(payment_type: PaymentType, price: float) -> float:
        match payment_type:
            case PaymentType.Monthly:
                return price * 12
            case PaymentType.Annually:
                return price
            case PaymentType.BiAnnually:
                return price * 2
            case PaymentType.Quarterly:
                return price * 4
            case _:
                assert_never(payment_type)

    @staticmethod
    def calculate_monthly_price(payment_type: PaymentType, price: float) -> float:
        match payment_type:
            case PaymentType.Monthly:
                return price
            case PaymentType.Annually:
                return price / 12
            case PaymentType.BiAnnually:
                return price / 6
            case PaymentType.Quarterly:
                return price / 3
            case _:
                assert_never(payment_type)

    def model_post_init(self, __context):
        self.annual_price = self.calculate_annual_price(self.payment_type, self.price)
        self.monthly_price = self.calculate_monthly_price(self.payment_type, self.price)


def _find_next_payment_iteratively(
    d: datetime.date,
    payment_type: PaymentType,
    today: Optional[datetime.date] = None,
    iter_count: int = 0,
):
    if today is None:
        today = datetime.date.today()

    match payment_type:
        case PaymentType.Monthly:
            gap = relativedelta.relativedelta(months=1)
            gap_aprox = datetime.timedelta(days=30)
        case PaymentType.Annually:
            gap = relativedelta.relativedelta(years=1)
            gap_aprox = datetime.timedelta(days=365)
        case PaymentType.BiAnnually:
            gap = relativedelta.relativedelta(months=6)
            gap_aprox = datetime.timedelta(days=180)
        case PaymentType.Quarterly:
            gap = relativedelta.relativedelta(months=3)
            gap_aprox = datetime.timedelta(days=89)
        case _:
            assert_never(payment_type)

    iter_count += 1
    time_from_today = d - today
    print(time_from_today, today)
    if 0 < time_from_today.days < gap_aprox.days:
        return d

    if iter_count > 1000:
        raise RuntimeError("did not converge")

    if time_from_today.days >= gap_aprox.days:
        return _find_next_payment_iteratively(d - gap, payment_type, today, iter_count)
    else:
        return _find_next_payment_iteratively(d + gap, payment_type, today, iter_count)


def calculate_next_payment(
    registered_payment_date: Optional[str],
    payment_type: PaymentType,
    today: Optional[datetime.date] = None,
) -> Optional[datetime.date]:

    if today is None:
        today = datetime.date.today()

    match payment_type:
        case PaymentType.Monthly:
            next_payment = today + relativedelta.relativedelta(months=1)
            return datetime.date(next_payment.year, next_payment.month, 1)

        case PaymentType.BiAnnually | PaymentType.Quarterly | PaymentType.Annually:
            if not registered_payment_date:
                return None
            y = today.year
            d, m = tuple(map(int, registered_payment_date.split("/")))
            assert 0 <= d <= 31
            assert 0 <= m <= 12
            candidate_date = datetime.date(y, m, d)
            return _find_next_payment_iteratively(candidate_date, payment_type, today)
        case _:
            assert_never(payment_type)


def get_google_sheet_data(
    spreadsheet_id: str, sheet_name: str, api_key: str
) -> Sequence[Payment]:
    """"""
    # Construct the URL for the Google Sheets API
    base_url = "https://sheets.googleapis.com/v4/spreadsheets"
    url = (
        f"{base_url}/{spreadsheet_id}/values/{sheet_name}!{COLUMN_START}1:{COLUMN_END}"
    )

    try:
        # Make a GET request to retrieve data from the Google Sheets API
        response = requests.get(
            url, timeout=REQUEST_TIMEOUT.seconds, params={"alt": "json", "key": api_key}
        )
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        data = response.json()
        rows = data["values"]
        _ = rows.pop(0)
        _rows = []
        for row in rows:
            print(dict(zip(Payment.model_fields, row)))
            _rows += [Payment(**dict(zip(Payment.model_fields, row)))]

        return _rows

    except requests.exceptions.RequestException as e:
        # Handle any errors that occur during the request
        print(f"An error occurred: {e}")
        return None


# https://docs.google.com/spreadsheets/d/1i1QSrzdVf--ei9XOUDk30WWwfT_ZGNg_4x5sFbFgMt0/edit?usp=sharing

payments = get_google_sheet_data(SHEET_ID, "Fixed & variable costs", API_KEY)

eval_date = datetime.date.today()
# next_payment_date = calculate_next_payment(None, PaymentType.Monthly, today)
date_range = pd.date_range(start=eval_date, periods=12, freq="ME")

total_payments = []
for curr_date in date_range:
    curr_date = curr_date.date()
    next_monthly_payment_date = calculate_next_payment(
        None, PaymentType.Monthly, curr_date
    )
    next_date_for_payment = [
        calculate_next_payment(row.starting_point, row.payment_type, curr_date)
        for row in payments
    ]
    next_total_payment = sum(
        p.price
        for p, d in zip(payments, next_date_for_payment)
        if d == next_monthly_payment_date
    )
    total_payments += [(next_monthly_payment_date, next_total_payment)]
df = pd.DataFrame(total_payments)
df2 = pl.DataFrame(payments).with_columns(
    next_date_for_payment=pl.Series(next_date_for_payment)
)

print(df)
