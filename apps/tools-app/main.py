from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_APPS = Path(__file__).parent.parent

pages = {
    "Finance": [
        st.Page(_APPS / "budget-app" / "app.py", title="Budget", icon="💰", url_path="budget"),
    ],
    "Energy": [
        st.Page(_APPS / "power-app" / "main.py", title="Power Usage", icon="⚡", url_path="power"),
    ],
    "Tools": [
        st.Page(_APPS / "transcribe" / "main.py", title="Transcribe", icon="🎙️", url_path="transcribe"),
    ],
}

pg = st.navigation(pages)
pg.run()
