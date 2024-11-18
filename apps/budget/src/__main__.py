import datetime
import sys
import polars as pl
import os
from src import payment
from src import schemas
from src import data_loader

SHEET_ID = os.getenv("SHEET_ID")
API_KEY = os.getenv("API_KEY")


if __name__ == "__main__":

    n_args = len(sys.argv)
    if n_args <= 2:
        _, arg = sys.argv

        if not arg:
            arg = """
            {
                "saldo": 72000,
                "monthly_salary": 44000,
                "additional_cost": 6000,
                "planned_projects": [
                    ["2024-12-01", 58000],
                    ["2025-02-01", 15000]
                ],
                "periods": 60
            }
            """
    else:
        raise ValueError(f"Expects only 0 or 1 arguments. Inputs are: {sys.argv}")

    config = schemas.PaymentConfig.model_validate_json(arg)

    eval_date = datetime.date.today()
    payments = data_loader.get_google_sheet_data(
        SHEET_ID, "Fixed & variable costs", API_KEY
    )
    next_date_for_payment = [
        payment.calculate_next_payment(row.starting_point, row.payment_type, eval_date)
        for row in payments
    ]

    payments_df = pl.DataFrame(payments).with_columns(
        next_date_for_payment=pl.Series(next_date_for_payment)
    )
    projects_df = pl.DataFrame(
        iter(config.planned_projects),
        schema=pl.Schema({"date": pl.Date, "project_cost": pl.Int64}),
    )

    def make_salary_sequence(monthly_salary: float, periods: int) -> list[float]:
        match monthly_salary:
            case float():
                return [monthly_salary] * periods

    df = (
        pl.DataFrame(
            payment.calculate_total_payments(
                eval_date=eval_date, payments=payments, monthly_periods=config.periods
            ),
            schema=pl.Schema({"date": pl.Date, "living_cost": pl.Float32}),
        )
        .join(other=projects_df, on=["date"], how="left")
        .with_columns(project_cost=pl.col("project_cost").fill_null(0))
        .with_columns(
            salary=pl.Series(
                make_salary_sequence(config.monthly_salary, config.periods)
            )
        )
        .with_columns(
            additional_cost=pl.Series([config.additional_cost] * config.periods)
        )
        .with_columns(
            delta=(
                pl.col("salary")
                - pl.col("project_cost")
                - pl.col("living_cost")
                - pl.col("additional_cost")
            )
        )
        .with_columns(saldo=config.saldo + pl.col("delta").cum_sum())
    )
    print(df.to_pandas())
