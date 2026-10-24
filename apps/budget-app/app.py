from collections.abc import Iterator
import os
import datetime
import pathlib
import dateutil
import pydantic
import polars as pl
import streamlit as st
from wiz.budget import data_loader, payment, schemas
from dotenv import load_dotenv

DEFAULT_STATE_FILE = pathlib.Path.home() / ".local" / "state" / "budget-app" / "state.json"


def state_file() -> pathlib.Path:
    """Persistent state path (survives reboots); override with BUDGET_STATE_FILE."""
    return pathlib.Path(os.environ.get("BUDGET_STATE_FILE", DEFAULT_STATE_FILE))


def save_state(interface: schemas.PaymentInterface) -> None:
    """Save the payment interface state to the persistent state file."""
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(interface.model_dump_json(indent=2))


def load_state() -> schemas.PaymentInterface | None:
    """Load the payment interface state from the persistent state file."""
    path = state_file()
    if path.exists():
        return schemas.PaymentInterface.model_validate_json(path.read_text())
    return None


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

    with st.expander("Table View"):
        st.write(pl.DataFrame(payments))

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


def _render_planned_projects(
    planned_projects: list[schemas.Record], num_projects: int
) -> list[schemas.Record]:
    """Render one input row per project, seeded from saved projects where available."""
    assert num_projects >= 0
    col1, col2, col3 = st.columns([2, 1, 1])

    default_date = datetime.date.today().replace(
        day=1
    ) + dateutil.relativedelta.relativedelta(months=1)
    projects = planned_projects[:num_projects] + [
        schemas.Record(amount=10_000.0, date=default_date)
        for _ in range(num_projects - len(planned_projects))
    ]

    records = []
    for i, project in enumerate(projects):
        with col1:
            _ = st.text_input(f"Project {i+1} Name", key=f"name_{i}")
        with col2:
            amount = st.number_input(
                f"Project {i+1} Amount",
                value=project.amount,
                step=1000.0,
                key=f"project_amount_{i}",
            )
        with col3:
            due_date = st.date_input(
                f"Project {i+1} Due Date",
                value=project.date,
                key=f"project_due_date_{i}",
            )
        if amount:
            records.append(schemas.Record(amount=amount, date=due_date))

    return records


def collect_payment_interface_inputs(
    previous_state: schemas.PaymentInterface | None = None,
) -> schemas.PaymentInterface:
    if previous_state:
        saldo = previous_state.saldo
        monthly_salary = previous_state.monthly_salary
        additional_cost = previous_state.additional_cost
        periods = previous_state.periods
        planned_projects = previous_state.planned_projects

    else:
        saldo = 30_000.0
        monthly_salary = 44_000.0
        additional_cost = 6_000.0
        periods = 12
        planned_projects = []

    # Streamlit App
    st.title("💰 Payment Interface App")
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            saldo = st.number_input(
                "💰 Current Balance (Saldo)", value=saldo, step=1_000.0
            )

        with col2:
            monthly_salary = st.number_input(
                "💵 Monthly Salary", value=monthly_salary, step=1_000.0
            )
        with col3:
            additional_cost = st.number_input(
                "💸 Additional Monthly Cost", value=additional_cost, step=500.0
            )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        periods = st.number_input(
            "📅 Number of Periods", value=periods, step=1, min_value=1
        )
    with col2:
        rundate = st.date_input("📆 Run Date", value=datetime.date.today())
    with col3:
        project_count = len(planned_projects) if planned_projects else 1
        num_projects: int = st.number_input(
            "📊 Planned Projects", value=project_count, step=1, min_value=0
        )

    st.title("📊 Planned Projects")
    planned_projects = _render_planned_projects(
        planned_projects=planned_projects, num_projects=num_projects
    )

    submit_col, save_col, _ = st.columns([1, 1, 4])
    submit = submit_col.button("Submit", key="submit_button")
    save = save_col.button(
        "💾 Save", key="save_button", help="Overwrite the saved session with these inputs"
    )
    if not (submit or save):
        return None

    try:
        # Validate using Pydantic
        payment_interface = schemas.PaymentInterface(
            saldo=saldo,
            monthly_salary=monthly_salary,
            additional_cost=additional_cost,
            planned_projects=planned_projects or [],
            periods=periods,
            rundate=rundate,
        )
    except pydantic.ValidationError as e:
        st.error("❌ Validation Error!")
        st.text(e)
        return None

    if save:
        save_state(payment_interface)
        st.success(f"💾 Saved to {state_file()}")
    if submit:
        st.success("✅ Payment Data Successfully loaded!")
        return payment_interface
    return None


def main() -> None:
    prev_state = load_state()
    interface = collect_payment_interface_inputs(previous_state=prev_state)
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
