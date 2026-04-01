"""Transport fradrag calculator — working days commuted per month."""

import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import holidays
import pandas as pd
import streamlit as st

_DATA_FILE = Path(__file__).parent / "vacation_days.json"


def _load_vacation() -> str:
    if _DATA_FILE.exists():
        saved = json.loads(_DATA_FILE.read_text())
        return "\n".join(saved)
    return ""


def _save_vacation(lines: list[str]) -> None:
    _DATA_FILE.write_text(json.dumps(sorted(lines)))


def working_days_in_month(year: int, month: int, vacation: set[date], wfh_days: int) -> int:
    dk_holidays = set(holidays.Denmark(years=year).keys())
    _, last_day = calendar.monthrange(year, month)
    count = 0
    current_week = None
    week_wfh = 0

    for day_num in range(1, last_day + 1):
        d = date(year, month, day_num)
        if d.weekday() >= 5:
            continue
        if d in dk_holidays or d in vacation:
            continue

        iso_week = d.isocalendar()[1]
        if iso_week != current_week:
            current_week = iso_week
            week_wfh = 0

        if week_wfh < wfh_days:
            week_wfh += 1
        else:
            count += 1

    return count


# Load saved vacation days into session state once
if "vacation_text" not in st.session_state:
    st.session_state.vacation_text = _load_vacation()

st.title("Transport Fradrag")
st.caption("Beregn arbejdsdage med transport pr. måned")

col1, col2 = st.columns(2)
with col1:
    year = st.number_input("År", min_value=2020, max_value=2030, value=date.today().year)
with col2:
    wfh_days = st.radio("Hjemmearbejdsdage pr. uge", options=[1, 2], horizontal=True)

st.subheader("Feriedage")
vacation_input = st.text_area(
    "Én dato eller interval pr. linje — YYYY-MM-DD eller YYYY-MM-DD:YYYY-MM-DD — gemmes automatisk",
    value=st.session_state.vacation_text,
    height=150,
    key="vacation_area",
)

# Parse and validate — single dates and ranges; weekends within ranges are ignored
vacation_dates: set[date] = set()
valid_lines: list[str] = []
for line in vacation_input.splitlines():
    line = line.strip()
    if not line:
        continue
    if ":" in line:
        parts = line.split(":", 1)
        try:
            start = date.fromisoformat(parts[0].strip())
            end = date.fromisoformat(parts[1].strip())
            if end < start:
                st.warning(f"Interval slut er før start: '{line}'")
                continue
            d = start
            while d <= end:
                if d.weekday() < 5:
                    vacation_dates.add(d)
                d += timedelta(days=1)
            valid_lines.append(line)
        except ValueError:
            st.warning(f"Ugyldigt intervalformat: '{line}' — brug YYYY-MM-DD:YYYY-MM-DD")
    else:
        try:
            vacation_dates.add(date.fromisoformat(line))
            valid_lines.append(line)
        except ValueError:
            st.warning(f"Ugyldigt datoformat: '{line}' — brug YYYY-MM-DD")

# Persist whenever the content changes
if valid_lines != st.session_state.vacation_text.splitlines():
    _save_vacation(valid_lines)
    st.session_state.vacation_text = "\n".join(sorted(valid_lines))

# Monthly table
rows = []
total = 0
for month in range(1, 13):
    count = working_days_in_month(int(year), month, vacation_dates, wfh_days)
    total += count
    rows.append({"Måned": date(int(year), month, 1).strftime("%B"), "Pendlerdage": count})

st.subheader(f"Oversigt — {year}")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.metric("Samlet pendlerdage", total)
