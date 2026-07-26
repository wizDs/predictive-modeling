# Whisper Transcriber - Opsummering af Setup

## Oversigt

En Streamlit-webapp til lydoptagelse og transskription med OpenAI Whisper.

---

## Python Dependencies

### `openai-whisper`
- **Hvad:** OpenAI's speech-to-text model
- **Hvorfor:** Kernen i appen - udfører selve transskriptionen af lyd til tekst
- **Modeller:** tiny, base, small, medium, large, turbo (større = mere præcis, men langsommere)

### `streamlit`
- **Hvad:** Web UI framework til Python
- **Hvorfor:** Gør det nemt at bygge interaktive webapps uden frontend-kode
- **Feature brugt:** `st.audio_input()` - native lydoptagelse direkte i browseren

---

## System Dependencies (via Homebrew)

### `ffmpeg`
- **Hvad:** Kommandolinjeværktøj til audio/video behandling
- **Hvorfor:** Whisper bruger ffmpeg internt til at læse og konvertere lydfiler
- **Fejl uden:** `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
- **Installation:** `brew install ffmpeg`

### `portaudio` (kun nødvendig med pyaudio)
- **Hvad:** Cross-platform audio I/O bibliotek
- **Hvorfor:** Kræves for at bygge `pyaudio` fra source
- **Fejl uden:** `fatal error: 'portaudio.h' file not found`
- **Installation:** `brew install portaudio`

---

## Problemer og Løsninger

### Problem 1: `streamlit-audiorecorder` + Python 3.13
```
ModuleNotFoundError: No module named 'audioop'
```
- **Årsag:** Python 3.13 fjernede det indbyggede `audioop` modul, som `pydub` (dependency af audiorecorder) bruger
- **Løsning:** Skiftede til Streamlit's native `st.audio_input()` widget i stedet

### Problem 2: pyaudio build fejl
```
fatal error: 'portaudio.h' file not found
```
- **Årsag:** `pyaudio` er en C-extension der kræver `portaudio` systembiblioteket
- **Løsning:** `brew install portaudio` før `uv sync`

### Problem 3: ffmpeg mangler
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```
- **Årsag:** Whisper kalder `ffmpeg` som subprocess for at loade lydfiler
- **Løsning:** `brew install ffmpeg`

---

## Endelig Dependency Liste

### `pyproject.toml`
```toml
dependencies = [
    "openai-whisper>=20250625",
    "streamlit>=1.40.0",
]
```

### System (macOS)
```bash
brew install ffmpeg
```

---

## Kørsel

```bash
cd apps/transcribe
uv sync
uv run streamlit run main.py
```

Åbn derefter http://localhost:8501 i browseren.

---

## Arkitektur

```
Browser (mikrofon) 
    ↓ st.audio_input()
Streamlit Server
    ↓ gem som .wav temp fil
Whisper Model
    ↓ ffmpeg læser fil
    ↓ model.transcribe()
Tekst output
    ↓ 
Browser (visning)
```

