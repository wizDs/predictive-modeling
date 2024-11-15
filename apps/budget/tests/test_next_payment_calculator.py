from typing import assert_never
from dateutil import relativedelta
import datetime
from src import payment

HALF_YEAR = relativedelta.relativedelta(months=6)
YEAR = relativedelta.relativedelta(months=12)
MONTH = relativedelta.relativedelta(months=1)
QUARTER = relativedelta.relativedelta(months=3)


def test_equal():
    today = datetime.date(2024, 3, 1)
    last_registered_payment_date = datetime.date(2024, 3, 1)
    for payment_type in payment.PaymentType:
        next_payment = payment._find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        match payment_type:
            case payment.PaymentType.BiAnnually:
                assert next_payment == last_registered_payment_date + HALF_YEAR
            case payment.PaymentType.Annually:
                assert next_payment == last_registered_payment_date + YEAR
            case payment.PaymentType.Monthly:
                assert next_payment == last_registered_payment_date + MONTH
            case payment.PaymentType.Quarterly:
                assert next_payment == last_registered_payment_date + QUARTER
            case _:
                assert_never(payment_type)


def test_just_payed():
    today = datetime.date(2024, 3, 16)
    last_registered_payment_date = datetime.date(2024, 3, 1)
    for payment_type in payment.PaymentType:
        next_payment = payment._find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        match payment_type:
            case payment.PaymentType.BiAnnually:
                assert next_payment == last_registered_payment_date + HALF_YEAR
            case payment.PaymentType.Annually:
                assert next_payment == last_registered_payment_date + YEAR
            case payment.PaymentType.Monthly:
                assert next_payment == last_registered_payment_date + MONTH
            case payment.PaymentType.Quarterly:
                assert next_payment == last_registered_payment_date + QUARTER
            case _:
                assert_never(payment_type)


def test_last_month_ago():
    today = datetime.date(2024, 3, 15)
    last_registered_payment_date = datetime.date(2024, 2, 2)
    paymemt_date_curr_month = datetime.date(
        today.year, today.month, last_registered_payment_date.day
    )

    for payment_type in payment.PaymentType:
        next_payment = payment._find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        print(next_payment)
        match payment_type:
            case payment.PaymentType.BiAnnually:
                assert next_payment == last_registered_payment_date + HALF_YEAR
            case payment.PaymentType.Annually:
                assert next_payment == last_registered_payment_date + YEAR
            case payment.PaymentType.Monthly:
                assert next_payment == paymemt_date_curr_month + MONTH
            case payment.PaymentType.Quarterly:
                assert next_payment == last_registered_payment_date + QUARTER
            case _:
                assert_never(payment_type)
