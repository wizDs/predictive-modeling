import datetime
from typing import NamedTuple, Optional, Sequence, assert_never, Iterator

from dateutil import relativedelta
import pandas as pd
from wiz.budget.schemas import PaymentType, Payment


def find_next_payment_iteratively(
    d: datetime.date,
    payment_type: PaymentType,
    today: Optional[datetime.date] = None,
    iter_count: int = 0,
):
    if today is None:
        today = datetime.date.today()

    gap = payment_type.relativedelta
    gap_aprox = payment_type.timedelta

    iter_count += 1
    time_from_today = d - today

    if time_from_today.days == 0:
        return d + gap
    if 0 < time_from_today.days < gap_aprox.days:
        return d

    if iter_count > 1000:
        raise RuntimeError("did not converge")

    if time_from_today.days >= gap_aprox.days:
        return find_next_payment_iteratively(d - gap, payment_type, today, iter_count)

    return find_next_payment_iteratively(d + gap, payment_type, today, iter_count)


def calculate_next_payment(
    registered_payment_date: Optional[str],
    payment_type: PaymentType,
    today: Optional[datetime.date] = None,
) -> datetime.date:

    if today is None:
        today = datetime.date.today()

    match payment_type:
        case PaymentType.MONTHLY:
            next_payment = today + relativedelta.relativedelta(months=1)
            return datetime.date(next_payment.year, next_payment.month, 1)

        case PaymentType.BIANNUALLY | PaymentType.QUARTERLY | PaymentType.ANNUALLY:
            if not registered_payment_date:
                raise ValueError(
                    "registered_payment_date must be set if payment type is not monthly"
                )
            y = today.year
            d, m = tuple(map(int, registered_payment_date.split("/")))
            assert 0 <= d <= 31
            assert 0 <= m <= 12
            candidate_date = datetime.date(y, m, d)
            return find_next_payment_iteratively(candidate_date, payment_type, today)
        case _:
            assert_never(payment_type)


class LivingCost(NamedTuple):
    payment_date: datetime.date
    total_payment: float


def calculate_total_payments(
    eval_date: datetime.date, payments: Sequence[Payment], monthly_periods: int = 12
) -> Iterator[LivingCost]:

    date_range = pd.date_range(start=eval_date, periods=monthly_periods, freq="ME")

    for curr_timestamp in date_range:
        curr_date = curr_timestamp.date()
        next_monthly_payment_date = calculate_next_payment(
            None, PaymentType.MONTHLY, curr_date
        )
        total_payment: float = 0.0
        for p in payments:
            match p.payment_type:
                case PaymentType.MONTHLY:
                    total_payment += p.price
                case (
                    PaymentType.QUARTERLY
                    | PaymentType.ANNUALLY
                    | PaymentType.BIANNUALLY
                ):
                    next_payment = calculate_next_payment(
                        p.starting_point, p.payment_type, curr_date
                    )
                    if next_payment == next_monthly_payment_date:
                        total_payment += p.price

        yield LivingCost(next_monthly_payment_date, total_payment)
