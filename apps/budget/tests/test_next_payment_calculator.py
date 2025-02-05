import datetime
from typing import assert_never
from src import payment
from src.schemas import PaymentType


def test_equal():
    today = datetime.date(2024, 3, 1)
    last_registered_payment_date = datetime.date(2024, 3, 1)
    for payment_type in PaymentType:
        next_payment = payment.find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        assert next_payment == last_registered_payment_date + payment_type.relativedelta


def test_just_payed():
    today = datetime.date(2024, 3, 16)
    last_registered_payment_date = datetime.date(2024, 3, 1)
    for payment_type in PaymentType:
        next_payment = payment.find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        assert next_payment == last_registered_payment_date + payment_type.relativedelta


def test_last_month():
    today = datetime.date(2024, 3, 15)
    last_registered_payment_date = datetime.date(2024, 2, 1)
    paymemt_date_curr_month = datetime.date(
        today.year, today.month, last_registered_payment_date.day
    )

    for payment_type in payment.PaymentType:
        next_payment = payment.find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        print(next_payment)
        if payment_type == PaymentType.MONTHLY:
            assert next_payment == paymemt_date_curr_month + payment_type.relativedelta
        else:
            assert (
                next_payment
                == last_registered_payment_date + payment_type.relativedelta
            )


def test_two_months():
    today = datetime.date(2025, 3, 15)
    last_registered_payment_date = datetime.date(2025, 1, 2)
    paymemt_date_curr_month = datetime.date(
        today.year, today.month, last_registered_payment_date.day
    )

    for payment_type in payment.PaymentType:
        next_payment = payment.find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        print(next_payment)
        if payment_type == PaymentType.MONTHLY:
            assert next_payment == paymemt_date_curr_month + payment_type.relativedelta
        else:
            assert (
                next_payment
                == last_registered_payment_date + payment_type.relativedelta
            )


def test_in_near_future():
    today = datetime.date(2025, 3, 15)
    last_registered_payment_date = datetime.date(2025, 4, 2)
    paymemt_date_curr_month = datetime.date(
        today.year, today.month, last_registered_payment_date.day
    )

    for payment_type in payment.PaymentType:
        next_payment = payment.find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        print(next_payment)
        if payment_type == PaymentType.MONTHLY:
            assert next_payment == paymemt_date_curr_month + payment_type.relativedelta
        else:
            assert next_payment == last_registered_payment_date


def test_in_far_future():
    today = datetime.date(2025, 2, 15)
    last_registered_payment_date = datetime.date(2025, 12, 1)
    paymemt_date_curr_month = datetime.date(
        today.year, today.month, last_registered_payment_date.day
    )

    for payment_type in payment.PaymentType:
        next_payment = payment.find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        print(next_payment)
        match payment_type:
            case payment.PaymentType.BIANNUALLY:
                assert next_payment == datetime.date(2025, 6, 1)
            case payment.PaymentType.ANNUALLY:
                assert next_payment == datetime.date(2025, 12, 1)
            case payment.PaymentType.MONTHLY:
                assert (
                    next_payment == paymemt_date_curr_month + payment_type.relativedelta
                )
            case payment.PaymentType.QUARTERLY:
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
        next_payment = payment.find_next_payment_iteratively(
            d=last_registered_payment_date,
            payment_type=payment_type,
            today=today,
        )
        print(next_payment)
        match payment_type:
            case payment.PaymentType.BIANNUALLY:
                assert next_payment == datetime.date(2025, 6, 1)
            case payment.PaymentType.ANNUALLY:
                assert next_payment == datetime.date(2025, 12, 1)
            case payment.PaymentType.MONTHLY:
                assert (
                    next_payment == paymemt_date_curr_month + payment_type.relativedelta
                )
            case payment.PaymentType.QUARTERLY:
                assert next_payment == datetime.date(2025, 3, 1)
            case _:
                assert_never(payment_type)
