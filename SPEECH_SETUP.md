# AutoType v1.02 — Speech Setup Guide

## Overview
AutoType supports **Speech-to-Text** (STT) via faster-whisper and **Text-to-Speech** (TTS) via Windows SAPI (pyttsx3).

---

## Requirements

### Core (already installed)
```
pip install flask requests sentence-transformers numpy
```

### Speech Dependencies
```
pip install faster-whisper pyttsx3
```

### FFmpeg (required for STT)
faster-whisper uses ffmpeg internally to decode browser audio (webm/opus format).

**Windows (via winget):**
```
winget install ffmpeg
```

**Windows (manual):**
1. Download from https://www.gyan.dev/ffmpeg/builds/
2. Extract and add the `bin` folder to your system PATH
3. Verify: `ffmpeg -version`

**Windows (via Chocolatey):**
```
choco install ffmpeg
```

---

## Speech-to-Text (STT) — faster-whisper

### First-Time Setup
1. Go to the **Speech** tab in AutoType
2. Select a Whisper model size:
   - `tiny` — fastest, least accurate (~39MB)
   - `base` — good balance (~74MB) **recommended**
   - `small` — better accuracy (~244MB)
   - `medium` — high accuracy (~769MB)
   - `large-v3` — best accuracy (~1.5GB)
3. Click **LOAD MODEL** — first run downloads the model from HuggingFace
4. Alternatively, set a **Local Path** to a pre-downloaded CTranslate2 model directory

### Using a Local Model
1. Download a CTranslate2 whisper model (e.g., from HuggingFace)
2. Enter the full path in **Local Path** field (e.g., `C:\Users\Drew\models\whisper-base-ct2`)
3. Click **LOAD MODEL**

### Device & Compute Type
- **Device**: `Auto` detects GPU/CPU. Use `CUDA` if you have an NVIDIA GPU with CUDA.
- **Compute**: `Auto` picks `float16` for CUDA, `int8` for CPU. `int8` is fastest on CPU.

### Model Persistence
- The whisper model stays loaded in memory between transcriptions (no reload delay)
- **Auto-unload**: Set to >0 minutes to automatically free VRAM/RAM after idle time
- **UNLOAD** button: Manually free the model from memory
- Neither the wake-word nor continuous modes unload the model

### Footer Buttons
Located in the bottom bar after "TriOptimum Neural Systems":
- **🎤 WAKE** — Wake word mode. Listens for the configured wake word, then records your command until silence timeout.
- **📡 LISTEN** — Continuous recording mode. Records everything from toggle-on to toggle-off. No timeout. Press again to stop and transcribe.

### Wake Word Mode
1. Set your wake word in the Speech tab (default: "hey autotype")
2. Set silence timeout (default: 30 seconds)
3. Click **🎤 WAKE** in the footer
4. Say the wake word → status shows "SPEAK NOW"
5. Speak your message → after silence timeout, it transcribes and (optionally) auto-sends
6. Returns to listening for wake word

### Continuous Mode
1. Click **📡 LISTEN** in the footer
2. Status shows "🔴 RECORDING" — mic is active
3. Speak as long as you want — no timeout
4. Click **📡 LISTEN** again to stop recording
5. Audio is transcribed and placed in the input box

### Auto-Send
When enabled in the Speech tab, transcribed text is automatically sent to the LLM without pressing SEND.

---

## Text-to-Speech (TTS) — Windows SAPI

### Setup
1. `pip install pyttsx3`
2. pyttsx3 uses Windows' built-in Speech API — no additional downloads needed

### Installing Additional Voices
1. Open Windows Settings → Time & Language → Speech
2. Under "Manage voices", click "Add voices"
3. Install desired language packs (includes TTS voices)
4. Third-party SAPI5 voices (purchased separately) also work
5. Restart AutoType after installing new voices

### Using TTS
1. Go to the **Speech** tab
2. Select a voice from the dropdown
3. Adjust **Rate** (speed) and **Volume**
4. Click **TEST VOICE** to preview
5. Enable **Auto-speak** to automatically speak every assistant response
6. Or use the **🔊 TTS** footer button to toggle TTS on/off at any time

### Footer Button
- **🔊 TTS** — Toggle TTS on/off. When on, assistant responses are spoken aloud after generation completes.

---

## Troubleshooting

### STT: "ffmpeg not found"
→ Install ffmpeg and ensure it's on your system PATH. Restart terminal/AutoType.

### STT: Model download stuck
→ Check internet connection. Or download manually and set Local Path.

### STT: CUDA out of memory
→ Use a smaller model (tiny/base) or switch Device to CPU.

### TTS: No voices in dropdown
→ Ensure pyttsx3 is installed. On Windows, SAPI voices should auto-detect.

### TTS: Voice sounds wrong
→ Try different voices. Adjust Rate slider. Some voices need specific rate ranges.

### Mic permission denied
→ Browser needs microphone permission. Click the lock icon in the address bar → allow mic.
→ If using HTTP (not HTTPS), Chrome blocks mic on non-localhost. Use `localhost:7865`.
