from datetime import datetime
import enum
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st
import whisper
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from storage import RecordingStorage, bucket_from_env, client_from_env, new_recording_id

load_dotenv()


class ModelType(enum.StrEnum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


_FFMPEG_MISSING_MESSAGE = (
    "**ffmpeg not found.** Whisper shells out to the `ffmpeg` CLI to decode audio, "
    "and it isn't installed (or isn't on PATH).\n\n"
    "Install it, then restart the app:\n"
    "- **Windows:** `choco install ffmpeg` (or `winget install ffmpeg`)\n"
    "- **macOS:** `brew install ffmpeg`\n"
    "- **Linux:** `apt install ffmpeg` (or your distro's package manager)"
)


@st.cache_resource
def load_model(model_type: ModelType) -> whisper.Whisper:
    """Load and cache the Whisper model."""
    return whisper.load_model(model_type)


_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


@st.cache_resource
def get_storage() -> RecordingStorage | None:
    """Connect to the shared MinIO container job_app also uses (see .env.example).

    History is a nice-to-have here, not core to transcription, so a missing/unreachable
    MinIO is not fatal -- unlike job_app, which hard-requires it.
    """
    try:
        return RecordingStorage(client_from_env(), bucket_from_env())
    except Exception:
        return None


# Page config
st.set_page_config(
    page_title="Whisper Transcriber",
    page_icon="🎙️",
    layout="centered",
)

# Custom CSS for a distinctive look
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }

    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }

    .subtitle {
        font-family: 'Outfit', sans-serif;
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .stSelectbox label {
        font-family: 'Outfit', sans-serif !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    .transcription-box {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        color: #f1f5f9;
        line-height: 1.8;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }

    .info-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(0, 212, 255, 0.1));
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    .info-card p {
        font-family: 'Outfit', sans-serif;
        color: #cbd5e1;
        margin: 0;
    }

    .stButton > button {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        background: linear-gradient(90deg, #7c3aed, #00d4ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.4) !important;
    }

    .stSpinner > div {
        border-color: #7c3aed !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00d4ff !important;
    }

    .stAudio {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("# 🎙️ Whisper Transcriber")
st.markdown(
    '<p class="subtitle">Record audio and transcribe with OpenAI Whisper</p>',
    unsafe_allow_html=True,
)

# Surface the ffmpeg dependency immediately on page load, rather than only once the user
# has recorded/uploaded audio and clicked Transcribe.
if shutil.which("ffmpeg") is None:
    st.error(_FFMPEG_MISSING_MESSAGE)

# Sidebar for model selection
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "Select Model",
        options=list(ModelType),
        format_func=lambda x: f"{x.value.capitalize()}",
        index=0,  # Default to tiny for speed
        help="Larger models are more accurate but slower",
    )
    assert model_choice is not None  # index=0 always selects a default

    st.markdown(
        """
        <div class="info-card">
        <p><strong>Model Guide:</strong></p>
        <p>• <strong>Tiny/Base:</strong> Fast, good for clear audio</p>
        <p>• <strong>Small/Medium:</strong> Balanced performance</p>
        <p>• <strong>Large/Turbo:</strong> Best accuracy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    language = st.selectbox(
        "Language",
        options=[
            "Auto-detect",
            "English",
            "Danish",
            "German",
            "Spanish",
            "French",
            "Japanese",
            "Chinese",
        ],
        index=0,
    )

    st.markdown("### 💾 History")
    storage = get_storage()
    save_to_history = st.checkbox(
        "Save recordings to history",
        value=storage is not None,
        disabled=storage is None,
        help="Stores the audio and transcript in MinIO (see .env.example) so you can revisit them later.",
    )
    if storage is None:
        st.caption(
            "MinIO not reachable -- history is disabled. Start it with `docker compose up -d` "
            "from `apps/tools-app/` (see `.env.example`), then reload."
        )

# Load model
with st.spinner(f"Loading {model_choice.value} model..."):
    model = load_model(model_choice)

# Main content - tabs for record vs upload vs history
tab_record, tab_upload, tab_history = st.tabs(["🎤 Record", "📁 Upload File", "📚 History"])

audio_data = None
audio_name: str

with tab_record:
    recorded_audio = st.audio_input(
        "Click to record from your microphone",
        key="audio_recorder",
    )
    if recorded_audio:
        audio_data = recorded_audio
        audio_name = "recording"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        key="audio_uploader",
    )
    if uploaded_file:
        audio_data = uploaded_file
        audio_name = Path(uploaded_file.name).stem

with tab_history:
    if storage is None:
        st.info("MinIO not reachable -- see the History note in the sidebar.")
    else:
        recordings = storage.list_recordings()
        if not recordings:
            st.markdown(
                '<div class="info-card"><p>No saved recordings yet. Transcribe something '
                "with history enabled to see it here.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            selected_recording = st.selectbox("Recording", recordings, key="history_recording")
            hist_audio = storage.load_audio(selected_recording)
            hist_transcript = storage.load_transcript(selected_recording)
            hist_meta = storage.load_meta(selected_recording)

            st.markdown(f"**Language:** {hist_meta.get('language', 'unknown')}  |  "
                        f"**Model:** {hist_meta.get('model', 'unknown')}")

            if hist_audio is not None:
                audio_filename = storage.audio_filename(selected_recording) or ""
                mime = _AUDIO_MIME.get(Path(audio_filename).suffix, "audio/wav")
                st.audio(hist_audio, format=mime)
                st.download_button(
                    "⬇️ Download Recording",
                    data=hist_audio,
                    file_name=Path(audio_filename).name or f"{selected_recording}.wav",
                    mime=mime,
                    key="history_download_audio",
                )

            st.markdown("### 📝 Transcription")
            st.markdown(f'<div class="transcription-box">{hist_transcript}</div>', unsafe_allow_html=True)
            st.code(hist_transcript, language=None)

# Process audio (from either source)
if audio_data is not None:
    # Display audio player
    st.markdown("### 🔊 Playback")
    st.audio(audio_data)

    # Download button (only for recordings)
    if audio_name == "recording":
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download Recording",
            data=audio_data,
            file_name=f"recording-{now}.wav",
            mime="audio/wav",
        )

    # Transcribe button
    if st.button("✨ Transcribe", use_container_width=True):
        if shutil.which("ffmpeg") is None:
            st.error(_FFMPEG_MISSING_MESSAGE)
        else:
            with st.spinner("Transcribing..."):
                # Determine file extension
                if hasattr(audio_data, "name"):
                    suffix = Path(audio_data.name).suffix or ".wav"
                else:
                    suffix = ".wav"

                # Save audio to temp file
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                    tmp_file.write(audio_data.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # Set language if specified
                    transcribe_options = {}
                    if language != "Auto-detect":
                        lang_map = {
                            "English": "en",
                            "Danish": "da",
                            "German": "de",
                            "Spanish": "es",
                            "French": "fr",
                            "Japanese": "ja",
                            "Chinese": "zh",
                        }
                        transcribe_options["language"] = lang_map.get(language)

                    # Transcribe
                    result = model.transcribe(tmp_path, **transcribe_options)

                    # Display results
                    st.markdown("### 📝 Transcription")

                    # Detected language
                    if "language" in result:
                        st.markdown(f"**Detected Language:** {result['language']}")

                    # Transcription text
                    st.markdown(
                        f'<div class="transcription-box">{result["text"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # Copy button
                    st.code(result["text"], language=None)

                    # Persist to the shared MinIO container, if enabled and reachable
                    if save_to_history and storage is not None:
                        recording_id = new_recording_id()
                        storage.save(
                            recording_id,
                            audio_data.getvalue(),
                            suffix,
                            result["text"],
                            result.get("language", "unknown"),
                            model_choice.value,
                        )
                        st.caption(f"💾 Saved to history as `{recording_id}`")

                finally:
                    # Cleanup temp file
                    Path(tmp_path).unlink(missing_ok=True)

else:
    st.markdown(
        """
        <div class="info-card">
        <p>👆 Use the <strong>Record</strong> tab to record from your microphone,</p>
        <p>or the <strong>Upload File</strong> tab to transcribe an existing audio file.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #64748b; font-family: Outfit, sans-serif;">'
    'Powered by <a href="https://github.com/openai/whisper" style="color: #7c3aed;">OpenAI Whisper</a>'
    "</p>",
    unsafe_allow_html=True,
)
