from datetime import datetime
import enum
import tempfile
from pathlib import Path

import streamlit as st
import whisper


class ModelType(enum.StrEnum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


@st.cache_resource
def load_model(model_type: ModelType) -> whisper.Whisper:
    """Load and cache the Whisper model."""
    return whisper.load_model(model_type)


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

# Load model
with st.spinner(f"Loading {model_choice.value} model..."):
    model = load_model(model_choice)

# Main content
st.markdown("### 🎤 Record Audio")

# Native Streamlit audio input
audio_data = st.audio_input(
    "Click to record from your microphone",
    key="audio_recorder",
)

# Process recorded audio
if audio_data is not None:
    # Display audio player
    st.markdown("### 🔊 Playback")
    st.audio(audio_data, format="audio/wav")

    # Download button
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="⬇️ Download Recording",
        data=audio_data,
        file_name=f"recording-{now}.wav",
        mime="audio/wav",
    )

    # Transcribe button
    if st.button("✨ Transcribe", use_container_width=True):
        with st.spinner("Transcribing..."):
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
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

            finally:
                # Cleanup temp file
                Path(tmp_path).unlink(missing_ok=True)

else:
    st.markdown(
        """
        <div class="info-card">
        <p>👆 Click the <strong>microphone button</strong> above to record audio.</p>
        <p>When done recording, click <strong>Transcribe</strong> to get your text.</p>
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
