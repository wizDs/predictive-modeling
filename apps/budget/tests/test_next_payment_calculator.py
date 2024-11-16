from typing import assert_never
from dateutil import relativedelta
import datetime
from src import payment
from src.schemas import Payment, PaymentType

HALF_YEAR = relativedelta.relativedelta(months=6)
YEAR = relativedelta.relativedelta(months=12)
MONTH = relativedelta.relativedelta(months=1)
QUARTER = relativedelta.relativedelta(months=3)


def test_equal():
    today = datetime.date(2024, 3, 1)
    last_registered_payment_date = datetime.date(2024, 3, 1)
    for payment_type in PaymentType:
        next_payment = payment._find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        match payment_type:
            case PaymentType.BiAnnually:
                assert next_payment == last_registered_payment_date + HALF_YEAR
            case PaymentType.Annually:
                assert next_payment == last_registered_payment_date + YEAR
            case PaymentType.Monthly:
                assert next_payment == last_registered_payment_date + MONTH
            case PaymentType.Quarterly:
                assert next_payment == last_registered_payment_date + QUARTER
            case _:
                assert_never(payment_type)


def test_just_payed():
    today = datetime.date(2024, 3, 16)
    last_registered_payment_date = datetime.date(2024, 3, 1)
    for payment_type in PaymentType:
        next_payment = payment._find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        match payment_type:
            case PaymentType.BiAnnually:
                assert next_payment == last_registered_payment_date + HALF_YEAR
            case PaymentType.Annually:
                assert next_payment == last_registered_payment_date + YEAR
            case PaymentType.Monthly:
                assert next_payment == last_registered_payment_date + MONTH
            case PaymentType.Quarterly:
                assert next_payment == last_registered_payment_date + QUARTER
            case _:
                assert_never(payment_type)


def test_last_month():
    today = datetime.date(2024, 3, 15)
    last_registered_payment_date = datetime.date(2024, 2, 1)
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
            case PaymentType.BiAnnually:
                assert next_payment == last_registered_payment_date + HALF_YEAR
            case PaymentType.Annually:
                assert next_payment == last_registered_payment_date + YEAR
            case PaymentType.Monthly:
                assert next_payment == paymemt_date_curr_month + MONTH
            case PaymentType.Quarterly:
                assert next_payment == last_registered_payment_date + QUARTER
            case _:
                assert_never(payment_type)


def test_two_months():
    today = datetime.date(2025, 3, 15)
    last_registered_payment_date = datetime.date(2025, 1, 2)
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


def test_in_near_future():
    today = datetime.date(2025, 3, 15)
    last_registered_payment_date = datetime.date(2025, 4, 2)
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
                assert next_payment == last_registered_payment_date
            case payment.PaymentType.Annually:
                assert next_payment == last_registered_payment_date
            case payment.PaymentType.Monthly:
                assert next_payment == paymemt_date_curr_month + MONTH
            case payment.PaymentType.Quarterly:
                assert next_payment == last_registered_payment_date
            case _:
                assert_never(payment_type)


def test_in_far_future():
    today = datetime.date(2025, 2, 15)
    last_registered_payment_date = datetime.date(2025, 12, 1)
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
                assert next_payment == datetime.date(2025, 6, 1)
            case payment.PaymentType.Annually:
                assert next_payment == datetime.date(2025, 12, 1)
            case payment.PaymentType.Monthly:
                assert next_payment == paymemt_date_curr_month + MONTH
            case payment.PaymentType.Quarterly:
                assert next_payment == datetime.date(2025, 3, 1)
            case _:
                assert_never(payment_type)


def test_in_far_past():
    today = datetime.date(2025, 2, 15)
    last_registered_payment_date = datetime.date(2023, 12, 1)
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
                assert next_payment == datetime.date(2025, 6, 1)
            case payment.PaymentType.Annually:
                assert next_payment == datetime.date(2025, 12, 1)
            case payment.PaymentType.Monthly:
                assert next_payment == paymemt_date_curr_month + MONTH
            case payment.PaymentType.Quarterly:
                assert next_payment == datetime.date(2025, 3, 1)
            case _:
                assert_never(payment_type)
