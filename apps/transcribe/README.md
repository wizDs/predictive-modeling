# Whisper Transcriber

A simple Streamlit app for recording and transcribing audio using OpenAI's Whisper model.

## Features

- 🎤 Record audio directly from your browser
- 🔊 Playback recorded audio before transcription
- ✨ Transcribe using various Whisper model sizes
- 🌍 Support for multiple languages with auto-detection
- 📋 Easy copy of transcribed text

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Make sure you have `ffmpeg` installed:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Usage

Run the Streamlit app:

```bash
uv run streamlit run main.py
```

Then:
1. Select a model size in the sidebar (smaller = faster, larger = more accurate)
2. Click "Start Recording" and speak
3. Click "Stop Recording" when done
4. Click "Transcribe" to get your text

## Model Sizes

| Model  | Speed   | Accuracy | VRAM   |
|--------|---------|----------|--------|
| tiny   | ~10x    | Good     | ~1 GB  |
| base   | ~7x     | Better   | ~1 GB  |
| small  | ~4x     | Good     | ~2 GB  |
| medium | ~2x     | Great    | ~5 GB  |
| large  | 1x      | Best     | ~10 GB |
| turbo  | ~8x     | Great    | ~6 GB  |

## Reference

Based on [OpenAI Whisper](https://github.com/openai/whisper).

