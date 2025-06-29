from operator import attrgetter
import os
import datetime
import uuid
import dateutil
import pydantic
import polars as pl
import streamlit as st
from wiz.budget import data_loader, payment, schemas
from dotenv import load_dotenv

load_dotenv()

SHEET_ID = os.environ["SHEET_ID"]
API_KEY = os.environ["API_KEY"]
COST_TABLE_NAME = "Fixed & variable costs"
INCOME_TABLE_NAME = "Indkomst"


def load_expected_payments_to_polars(
    *,
    interface: schemas.PaymentInterface,
    google_sheet_id: str,
    table_name: str,
    api_key: str,
) -> pl.DataFrame:
    eval_date = interface.rundate or datetime.date.today()
    payments = data_loader.get_payment_data(
        spreadsheet_id=google_sheet_id,
        sheet_name=table_name,
        api_key=api_key,
        column_start="A",
        column_end="I",
    )
    return pl.DataFrame(
        payment.calculate_total_payments(
            eval_date=eval_date,
            payments=payments,
            monthly_periods=interface.periods,
        ),
        schema=pl.Schema({"date": pl.Date, "living_cost": pl.Float32}),
    )


def load_expected_income_to_polars(
    *,
    interface: schemas.PaymentInterface,
    google_sheet_id: str,
    table_name: str,
    api_key: str,
) -> pl.DataFrame:
    income_data = data_loader.get_income_data(
        spreadsheet_id=google_sheet_id,
        sheet_name=table_name,
        api_key=api_key,
        column_start="B",
        column_end="D",
    )
    return income_data


def collect_payment_interface_inputs() -> schemas.PaymentInterface:
    # Streamlit App
    st.title("💰 Payment Interface App")
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            saldo = st.number_input(
                "💰 Current Balance (Saldo)", value=30_000.0, step=1_000.0
            )

        with col2:
            monthly_salary = st.number_input(
                "💵 Monthly Salary", value=44_000.0, step=1_000.0
            )
        with col3:
            additional_cost = st.number_input(
                "💸 Additional Monthly Cost", value=6_000.0, step=500.0
            )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        periods = st.number_input("📅 Number of Periods", value=12, step=1, min_value=1)
    with col2:
        rundate = st.date_input("📆 Run Date", value=datetime.date.today())
    with col3:
        num_projects: int = st.number_input(
            "📊 Planned Projects", value=1, step=1, min_value=0
        )

    st.title("📊 Planned Projects")
    planned_projects = []

    for i in range(num_projects):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            _ = st.text_input(f"Project {i+1} Name", key=f"name_{i}")
        with col2:
            amount = st.number_input(
                f"Project {i+1} Amount",
                value=10_000.0,
                step=1000.0,
                key=str(uuid.uuid4().hex),
            )
        with col3:
            due_date = st.date_input(
                f"Project {i+1} Due Date",
                value=datetime.date(
                    year=datetime.date.today().year,
                    month=datetime.date.today().month,
                    day=1,
                )
                + dateutil.relativedelta.relativedelta(months=i + 1),
                key=str(uuid.uuid4().hex),
            )
        if amount:
            planned_projects.append(schemas.Record(amount=amount, date=due_date))

    # Submit Button
    if st.button("Submit", key="submit_button"):
        try:
            # Validate using Pydantic
            payment_interface = schemas.PaymentInterface(
                saldo=saldo,
                monthly_salary=monthly_salary,
                additional_cost=additional_cost,
                planned_projects=planned_projects,
                periods=periods,
                rundate=rundate,
            )
            st.success("✅ Payment Data Successfully loaded!")
            return payment_interface

        except pydantic.ValidationError as e:
            st.error("❌ Validation Error!")
            st.text(e)
            return None


def main() -> None:
    interface = collect_payment_interface_inputs()
    if interface:
        calculate(interface)


def calculate(payment_interface: schemas.PaymentInterface) -> None:

    total_payments_df = load_expected_payments_to_polars(
        interface=payment_interface,
        google_sheet_id=SHEET_ID,
        table_name=COST_TABLE_NAME,
        api_key=API_KEY,
    )

    projects_df = (
        pl.DataFrame(
            iter(payment_interface.planned_projects),
            schema=pl.Schema(
                {
                    "description": pl.Utf8,
                    "date": pl.Date,
                    "project_cost": pl.Int64,
                }
            ),
        )
        .with_columns(
            pl.when(pl.col("date").dt.day() == 1)
            .then(
                pl.col("date")
            )  # Keep the date unchanged if it's the 1st of the month
            .otherwise(
                pl.col("date").dt.offset_by("1mo").dt.truncate("1mo")
            )  # Move to next month's start
            .alias("date")
        )
        .group_by("date")
        .agg(pl.sum("project_cost"))
    )

    df = (
        total_payments_df.join(other=projects_df, on=["date"], how="left")
        .with_columns(project_cost=pl.col("project_cost").fill_null(0))
        .with_columns(salary=pl.lit(payment_interface.monthly_salary))
        .with_columns(additional_cost=pl.lit(payment_interface.additional_cost))
        .with_columns(
            delta=(
                pl.col("salary")
                - pl.col("project_cost")
                - pl.col("living_cost")
                - pl.col("additional_cost")
            )
        )
        .with_columns(saldo=payment_interface.saldo + pl.col("delta").cum_sum())
        .with_columns(date=pl.col("date").cast(dtype=pl.Utf8()))
    )
    st.write(df)


if __name__ == "__main__":
    main()
