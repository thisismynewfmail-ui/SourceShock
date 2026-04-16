"""
AutoType v1.02 — Speech Engine
STT: faster-whisper (persistent model, configurable unload timer)
TTS: pyttsx3 / Windows SAPI (voice selection, WAV output)

Requirements:
  pip install faster-whisper pyttsx3
  ffmpeg must be on PATH (faster-whisper uses it to decode audio)
"""
import os, json, time, threading, tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "speech")
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(DATA_DIR, "speech_config.json")

# ══════════════════════════════════════════════════════════════════
# STT — faster-whisper (persistent model)
# ══════════════════════════════════════════════════════════════════
_whisper_model = None
_whisper_lock = threading.Lock()
_whisper_last_use = 0.0
_unload_timer = None

def _get_whisper(model_size="base", model_path="", device="auto", compute_type="auto"):
    global _whisper_model, _whisper_last_use
    with _whisper_lock:
        if _whisper_model is not None:
            _whisper_last_use = time.time()
            return _whisper_model
        from faster_whisper import WhisperModel
        path = model_path.strip() if model_path.strip() else model_size
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
                print("[STT] torch unavailable, using CPU")
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        print(f"[STT] Loading whisper: {path} device={device} compute={compute_type}")
        _whisper_model = WhisperModel(path, device=device, compute_type=compute_type)
        _whisper_last_use = time.time()
        print("[STT] Model ready")
        return _whisper_model

def preload_whisper(config):
    """Pre-load model so first transcription is fast."""
    try:
        _get_whisper(
            model_size=config.get("whisper_model", "base"),
            model_path=config.get("whisper_model_path", ""),
            device=config.get("whisper_device", "auto"),
            compute_type=config.get("whisper_compute", "auto")
        )
        return True
    except Exception as e:
        print(f"[STT] Preload failed: {e}")
        return False

def unload_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            print("[STT] Model unloaded")
            return True
    return False

def whisper_loaded():
    return _whisper_model is not None

def whisper_info():
    return {
        "loaded": _whisper_model is not None,
        "last_use": _whisper_last_use,
        "idle_seconds": int(time.time() - _whisper_last_use) if _whisper_last_use > 0 else 0
    }

def _start_unload_timer(minutes):
    global _unload_timer
    if _unload_timer:
        _unload_timer.cancel()
    if minutes <= 0:
        return
    def _check():
        idle = time.time() - _whisper_last_use
        if idle >= minutes * 60 and _whisper_model is not None:
            print(f"[STT] Auto-unloading after {minutes}min idle")
            unload_whisper()
    _unload_timer = threading.Timer(minutes * 60, _check)
    _unload_timer.daemon = True
    _unload_timer.start()

def transcribe_audio(audio_bytes, content_type, config):
    """
    Transcribe audio bytes using faster-whisper.
    Browser sends webm/opus; faster-whisper uses ffmpeg to decode any format.
    """
    global _whisper_last_use
    model = _get_whisper(
        model_size=config.get("whisper_model", "base"),
        model_path=config.get("whisper_model_path", ""),
        device=config.get("whisper_device", "auto"),
        compute_type=config.get("whisper_compute", "auto")
    )
    # Pick extension from content type so ffmpeg handles it correctly
    ct = (content_type or "").lower()
    if "wav" in ct: ext = ".wav"
    elif "ogg" in ct: ext = ".ogg"
    elif "mp4" in ct or "m4a" in ct: ext = ".m4a"
    else: ext = ".webm"

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=DATA_DIR)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        segments, info = model.transcribe(
            tmp.name,
            language=config.get("language", None) or None,
            beam_size=config.get("beam_size", 5),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=600, speech_pad_ms=300)
        )
        text = " ".join([seg.text.strip() for seg in segments])
        _whisper_last_use = time.time()
        auto_min = config.get("auto_unload_minutes", 0)
        if auto_min > 0:
            _start_unload_timer(auto_min)
        return text.strip()
    finally:
        try: os.unlink(tmp.name)
        except: pass

# ══════════════════════════════════════════════════════════════════
# TTS — subprocess-isolated pyttsx3 (prevents COM deadlock in Flask)
# ══════════════════════════════════════════════════════════════════
_TTS_SCRIPT = os.path.join(DATA_DIR, "_tts_worker.py")

def _ensure_tts_script():
    """Write the TTS worker script if it doesn't exist or is outdated."""
    code = '''import sys, json, pyttsx3
cfg = json.loads(sys.argv[1])
engine = pyttsx3.init()
vid = cfg.get("voice", "")
if vid:
    engine.setProperty("voice", vid)
engine.setProperty("rate", int(cfg.get("rate", 170)))
engine.setProperty("volume", min(1.0, max(0.0, int(cfg.get("volume", 100)) / 100.0)))
outfile = cfg["outfile"]
engine.save_to_file(cfg["text"], outfile)
engine.runAndWait()
print("OK")
'''
    with open(_TTS_SCRIPT, "w") as f:
        f.write(code)

def get_tts_voices():
    """List available TTS voices (runs in subprocess to avoid COM issues)."""
    import subprocess, sys
    code = 'import json,pyttsx3;e=pyttsx3.init();print(json.dumps([{"id":v.id,"name":v.name} for v in e.getProperty("voices")]))'
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return json.loads(r.stdout.strip())
    except Exception as e:
        print(f"[TTS] Error listing voices: {e}")
    return []

def speak_to_wav(text, config):
    """Generate WAV via pyttsx3 in a subprocess (prevents COM threading crash)."""
    import subprocess, sys
    _ensure_tts_script()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=DATA_DIR)
    tmp.close()
    params = json.dumps({
        "text": text,
        "voice": config.get("tts_voice", ""),
        "rate": config.get("tts_rate", 170),
        "volume": config.get("tts_volume", 100),
        "outfile": tmp.name
    })
    try:
        r = subprocess.run(
            [sys.executable, _TTS_SCRIPT, params],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0:
            print(f"[TTS] Subprocess error: {r.stderr}")
            return None
        with open(tmp.name, 'rb') as f:
            wav_data = f.read()
        if len(wav_data) < 100:
            print("[TTS] WAV too small, generation likely failed")
            return None
        return wav_data
    except subprocess.TimeoutExpired:
        print("[TTS] Subprocess timed out")
        return None
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return None
    finally:
        try: os.unlink(tmp.name)
        except: pass

# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "whisper_model": "base",
    "whisper_model_path": "",
    "whisper_device": "auto",
    "whisper_compute": "auto",
    "language": "",
    "beam_size": 5,
    "wake_word": "hey autotype",
    "wake_timeout": 30,
    "auto_unload_minutes": 0,
    "auto_send": True,
    "tts_voice": "",
    "tts_rate": 170,
    "tts_volume": 100,
    "tts_auto_speak": False,
}

def load_speech_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg.update(json.load(f))
        except: pass
    return cfg

def save_speech_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
