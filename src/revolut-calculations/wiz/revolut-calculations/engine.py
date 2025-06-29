import math
import functools
import os
import enum
import polars as pl


class InterestType(enum.StrEnum):
    ANNUAL = "annual"
    MONTHLY = "monthly"


class RevolutPackage(enum.StrEnum):
    STANDARD = "standard"
    PLUS = "plus"
    PREMIUM = "premium"
    METAL = "metal"
    ULTRA = "ultra"

    @property
    def cost(self) -> float:
        match self:
            case RevolutPackage.STANDARD:
                return 0.00
            case RevolutPackage.PLUS:
                return 32.99
            case RevolutPackage.PREMIUM:
                return 74.99
            case RevolutPackage.METAL:
                return 129.99
            case RevolutPackage.ULTRA:
                return 450.00


def calculate_monthly_revolut_interest(
    *,
    amount: float,
    interest_rate: float,
    interest_type: InterestType = InterestType.ANNUAL,
    monthly_cost: float = 0.0,
) -> float:
    match interest_type:
        case InterestType.ANNUAL:
            interest_rate = annual_interest_rate_to_monthly_interest_rate(
                annual_interest_rate_pct=interest_rate
            )
        case InterestType.MONTHLY:
            ...
    return amount * (interest_rate / 100) - monthly_cost


def annual_interest_rate_to_monthly_interest_rate(
    *,
    annual_interest_rate_pct: float,
) -> float:
    return ((1 + annual_interest_rate_pct / 100) ** (1 / 12) - 1) * 100


if __name__ == "__main__":
    print(annual_interest_rate_to_monthly_interest_rate(annual_interest_rate_pct=3.5))
    assert math.isclose(
        calculate_monthly_revolut_interest(
            amount=100_000,
            interest_rate=1.5,
            monthly_cost=1000,
            interest_type=InterestType.MONTHLY,
        ),
        500,
    )
    print(
        calculate_monthly_revolut_interest(
            amount=100_000,
            interest_rate=1.23,
            interest_type=InterestType.ANNUAL,
            monthly_cost=RevolutPackage.STANDARD.cost,
        )
    )
    print(
        calculate_monthly_revolut_interest(
            amount=100_000,
            interest_rate=1.83,
            interest_type=InterestType.ANNUAL,
            monthly_cost=RevolutPackage.PREMIUM.cost,
        )
    )

    def compound_monthly_interest(
        *,
        amount: float,
        interest_rate: float,
        interest_type: InterestType = InterestType.ANNUAL,
        monthly_cost: float,
        n: int = 12,
        period: int = 1,
    ) -> float:
        interest = calculate_monthly_revolut_interest(
            amount=amount,
            interest_rate=interest_rate,
            monthly_cost=monthly_cost,
            interest_type=interest_type,
        )
        if period == n:
            return amount + interest
        return compound_monthly_interest(
            amount=amount + interest,
            interest_rate=interest_rate,
            monthly_cost=monthly_cost,
            interest_type=interest_type,
            n=n,
            period=period + 1,
        )

    print(
        compound_monthly_interest(
            amount=100_000,
            interest_rate=3.0,
            monthly_cost=RevolutPackage.STANDARD.cost,
            interest_type=InterestType.ANNUAL,
            n=10 * 12,
        )
    )
    print("Hello, World!")
