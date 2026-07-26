from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")

_HERE = Path(__file__).parent
_APPS = _HERE.parent

pages = {
    "Skat": [
        st.Page(
            _HERE / "transport_fradrag" / "main.py",
            title="Transport Fradrag",
            icon="🚌",
            url_path="transport-fradrag",
        ),
        st.Page(
            _HERE / "skat_app" / "main.py",
            title="Skatteberegner",
            icon="🇩🇰",
            url_path="skatteberegner",
        ),
    ],
    "Finance": [
        st.Page(
            _APPS / "budget-app" / "app.py",
            title="Budget",
            icon="💰",
            url_path="budget",
        ),
    ],
    "Energy": [
        st.Page(
            _APPS / "power-app" / "power-app.py",
            title="Power Usage",
            icon="⚡",
            url_path="power",
        ),
    ],
    "Tools": [
        st.Page(
            _APPS / "transcribe" / "main.py",
            title="Transcribe",
            icon="🎙️",
            url_path="transcribe",
        ),
        st.Page(
            _HERE / "job_app" / "main.py",
            title="Job Application",
            icon="📄",
            url_path="job-application",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()
