import datetime
import sys
import os
import polars as pl
from wiz.budget import payment
from wiz.budget import schemas
from wiz.budget import data_loader

SHEET_ID = os.environ["SHEET_ID"]
API_KEY = os.environ["API_KEY"]


def _make_salary_sequence(monthly_salary: float, periods: int) -> list[float]:
    match monthly_salary:
        case float():
            return [monthly_salary] * periods


def main(payment_interface: schemas.PaymentInterface) -> None:

    eval_date = payment_interface.rundate or datetime.date.today()
    payments = data_loader.get_google_sheet_data(
        SHEET_ID, "Fixed & variable costs", API_KEY
    )
    with pl.Config(tbl_cols=-1, tbl_rows=-1):
        print(pl.DataFrame(payments))
    total_payments_df = pl.DataFrame(
        payment.calculate_total_payments(
            eval_date=eval_date,
            payments=payments,
            monthly_periods=payment_interface.periods,
        ),
        schema=pl.Schema({"date": pl.Date, "living_cost": pl.Float32}),
    )
    projects_df = (
        pl.DataFrame(
            iter(payment_interface.planned_projects),
            schema=pl.Schema({"date": pl.Date, "project_cost": pl.Int64}),
        )
        .group_by("date")
        .agg(pl.sum("project_cost"))
    )

    df = (
        total_payments_df.join(other=projects_df, on=["date"], how="left")
        .with_columns(project_cost=pl.col("project_cost").fill_null(0))
        .with_columns(
            salary=pl.Series(
                _make_salary_sequence(
                    payment_interface.monthly_salary, payment_interface.periods
                )
            )
        )
        .with_columns(
            additional_cost=pl.Series(
                [payment_interface.additional_cost] * payment_interface.periods
            )
        )
        .with_columns(
            delta=(
                pl.col("salary")
                - pl.col("project_cost")
                - pl.col("living_cost")
                - pl.col("additional_cost")
            )
        )
        .with_columns(saldo=payment_interface.saldo + pl.col("delta").cum_sum())
    )
    print(df.to_pandas())


if __name__ == "__main__":
    DEFAULT_ARG = """
    {
        "saldo": 47342,
        "monthly_salary": 44000,
        "additional_cost": 6000,
        "planned_projects": [
            ["2025-04-01", 20000],
            ["2025-03-01", 20500],
            ["2025-03-01", 10500]
        ],
        "periods": 24,
        "rundate": "2025-02-01"
    }
    """
    n_args = len(sys.argv)
    if n_args == 1:
        ARG = DEFAULT_ARG
    elif n_args == 2:
        _, ARG = sys.argv

        if not ARG:
            ARG = DEFAULT_ARG
    else:
        raise ValueError(f"Expects only 0 or 1 arguments. Inputs are: {sys.argv}")

    interface = schemas.PaymentInterface.model_validate_json(ARG)
    main(interface)
