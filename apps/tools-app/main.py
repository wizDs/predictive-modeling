from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
    ],
}

pg = st.navigation(pages)
pg.run()
