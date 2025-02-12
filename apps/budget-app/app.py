import os
import datetime
import pydantic
import polars as pl
import streamlit as st
from dateutil import relativedelta
from wiz.budget import data_loader, payment, schemas
from dotenv import load_dotenv

load_dotenv()

SHEET_ID = os.environ["SHEET_ID"]
API_KEY = os.environ["API_KEY"]
TABLE_NAME = "Fixed & variable costs"


def load_expected_payments_to_polars(
    *,
    payment_interface: schemas.PaymentInterface,
    google_sheet_id: str,
    table_name: str,
    api_key: str,
) -> pl.DataFrame:
    eval_date = payment_interface.rundate or datetime.date.today()
    payments = data_loader.get_google_sheet_data(google_sheet_id, table_name, api_key)
    return pl.DataFrame(
        payment.calculate_total_payments(
            eval_date=eval_date,
            payments=payments,
            monthly_periods=payment_interface.periods,
        ),
        schema=pl.Schema({"date": pl.Date, "living_cost": pl.Float32}),
    )


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
        name = st.text_input(f"Project {i+1} Name", key=f"name_{i}")
    with col2:
        amount = st.number_input(
            f"Project {i+1} Amount", value=10_000.0, step=1000.0, key=f"amount_{i}"
        )
    with col3:
        due_date = st.date_input(f"Project {i+1} Due Date", key=f"date_{i}")
    if amount:
        planned_projects.append(
            schemas.Record(description=name, price=amount, due_date=due_date)
        )

# Submit Button
if st.button("Submit"):
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
        total_payments_df = load_expected_payments_to_polars(
            payment_interface=payment_interface,
            google_sheet_id=SHEET_ID,
            table_name=TABLE_NAME,
            api_key=API_KEY,
        )

        projects_df = (
            pl.DataFrame(
                iter(payment_interface.planned_projects),
                schema=pl.Schema(
                    {"description": pl.Utf8, "date": pl.Date, "project_cost": pl.Int64}
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
        )
        st.write(df)

    except pydantic.ValidationError as e:
        st.error("❌ Validation Error!")
        st.text(e)
