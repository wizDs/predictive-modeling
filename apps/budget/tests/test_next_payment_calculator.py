from dateutil import relativedelta
import datetime
from src import payment


def test_biannually():
    today = datetime.date(2024, 3, 16)
    last_registered_payment_date = datetime.date(2024, 3, 16)
    next_payment = payment._find_next_payment_iteratively(
        d=last_registered_payment_date,
        payment_type=payment.PaymentType.BiAnnually,
        today=today,
    )
    assert next_payment == today + relativedelta.relativedelta(months=6)
