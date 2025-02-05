import datetime
import enum
from typing import Any, Literal, NamedTuple, Optional, Sequence, assert_never
import pydantic
from dateutil import relativedelta


class PaymentType(enum.StrEnum):
    MONTHLY = "Måned"
    ANNUALLY = "År"
    QUARTERLY = "Kvartal"
    BIANNUALLY = "Halvårligt"

    @property
    def months(self) -> int:
        match self:
            case self.MONTHLY:
                return 1
            case self.ANNUALLY:
                return 12
            case self.QUARTERLY:
                return 3
            case self.BIANNUALLY:
                return 6
            case _:
                assert_never(self)

    @property
    def days(self) -> int:
        match self:
            case PaymentType.MONTHLY:
                return 30
            case PaymentType.ANNUALLY:
                return 365
            case PaymentType.BIANNUALLY:
                return 180
            case PaymentType.QUARTERLY:
                return 90
            case _:
                assert_never(self)

    @property
    def relativedelta(self) -> relativedelta.relativedelta:
        return relativedelta.relativedelta(months=self.months)

    @property
    def timedelta(self) -> datetime.timedelta:
        return datetime.timedelta(days=self.days)


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
        return price * payment_type.months

    @staticmethod
    def calculate_monthly_price(payment_type: PaymentType, price: float) -> float:
        return price / payment_type.months

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


class PaymentInterface(pydantic.BaseModel):
    saldo: float
    monthly_salary: float  # | ExpectedSalaryRecords[Record]
    additional_cost: float
    planned_projects: Sequence[Record]
    periods: int
    rundate: Optional[datetime.date] = None
