from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")

_HERE = Path(__file__).parent

pages = {
    "Finance": [
        st.Page(
            _HERE / "budget-app" / "app.py",
            title="Budget",
            icon="💰",
            url_path="budget",
        ),
    ],
    "Energy": [
        st.Page(
            _HERE / "power-app" / "power-app.py",
            title="Power Usage",
            icon="⚡",
            url_path="power",
        ),
    ],
    "Tools": [
        st.Page(
            _HERE / "transcribe" / "main.py",
            title="Transcribe",
            icon="🎙️",
            url_path="transcribe",
        ),
        st.Page(
            _HERE / "job-app" / "main.py",
            title="Job Application",
            icon="📄",
            url_path="job-application",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()
