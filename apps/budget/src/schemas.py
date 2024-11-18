import datetime
import enum
from operator import attrgetter
from typing import Any, Callable, Literal, NamedTuple, Optional, Sequence, assert_never

import pandas as pd
import pydantic


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


class Record(NamedTuple):
    date: datetime.date
    price: float


# class ExpectedSalaryRecords(pydantic.BaseModel):
#     records: Sequence[Record]

#     def monthly_salary_stream(self, periods: int) -> Sequence[float]:
#         ordered_records = sorted(self.records, key=attrgetter("date"))
#         for i in range(ordered_records):
#             if i < len(self.records):
#                 date_range = pd.date_range(
#                     start=ordered_records[i].date,
#                     end=ordered_records[i + 1].date,
#                     freq="MS",
#                     inclusive="left",
#                 )
#             else:
#                 pd.date_range(
#                     start=ordered_records[i].price,
#                     periods=ordered_records[0].date - ordered_records[i].date,
#                     freq="MS",
#                 )
#                 [ordered_records[i].price] * periods

#         self.records
#         return self.records


class PaymentConfig(pydantic.BaseModel):
    saldo: float
    monthly_salary: float  # | ExpectedSalaryRecords[Record]
    additional_cost: float
    planned_projects: Sequence[Record]
    periods: int
