import streamlit as st
import pydantic
import datetime
from typing import Sequence, Optional
from wiz.budget.schemas import PaymentInterface


# Define Pydantic Model
class Record(pydantic.BaseModel):
    name: str
    amount: float
    due_date: datetime.date


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

col1, col2 = st.columns([2, 1])
with col1:
    periods = st.number_input("📅 Number of Periods", value=12, step=1, min_value=1)
with col2:
    rundate = st.date_input("📆 Run Date", value=datetime.date.today())


# Planned Projects - Dynamic Inputs
st.subheader("📊 Planned Projects")
planned_projects = []
num_projects = st.number_input("How many projects?", value=1, step=1, min_value=0)

for i in range(num_projects):
    st.write(f"### Project {i+1}")
    name = st.text_input(f"Project {i+1} Name", key=f"name_{i}")
    amount = st.number_input(
        f"Project {i+1} Amount", value=100.0, step=50.0, key=f"amount_{i}"
    )
    due_date = st.date_input(f"Project {i+1} Due Date", key=f"date_{i}")
    if name:
        planned_projects.append(Record(name=name, amount=amount, due_date=due_date))

# Submit Button
if st.button("Submit"):
    try:
        # Validate using Pydantic
        payment_data = PaymentInterface(
            saldo=saldo,
            monthly_salary=monthly_salary,
            additional_cost=additional_cost,
            planned_projects=planned_projects,
            periods=periods,
            rundate=rundate,
        )
        st.success("✅ Payment Data Successfully Validated!")
        st.json(payment_data.model_dump())  # Display structured data

    except pydantic.ValidationError as e:
        st.error("❌ Validation Error!")
        st.text(e)
