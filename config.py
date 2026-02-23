"""Configuration for AI Avatar Agent — reads from .env file."""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── LiveKit ──────────────────────────────────────────────────────────────────
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# ── LLM ──────────────────────────────────────────────────────────────────────
SELFHOSTED_LLM_URL = os.getenv("SELFHOSTED_LLM_URL", "http://localhost:11434/v1")
SELFHOSTED_LLM_MODEL = os.getenv("SELFHOSTED_LLM_MODEL", "qwen2.5:7b")
SELFHOSTED_LLM_API_KEY = os.getenv("SELFHOSTED_LLM_API_KEY", "not-needed")

# ── VibeVoice TTS ─────────────────────────────────────────────────────────────
VIBEVOICE_TTS_URL = os.getenv("VIBEVOICE_TTS_URL", "http://localhost:9000")
VIBEVOICE_VOICE = os.getenv("VIBEVOICE_VOICE", "en-Carter_man")

# ── MuseTalk ──────────────────────────────────────────────────────────────────
MUSETALK_SERVER_URL = os.getenv("MUSETALK_SERVER_URL", "http://localhost:7860")
AVATAR_IMAGE = os.getenv("AVATAR_IMAGE", "/workspace/avatar.jpg")

# ── STT ──────────────────────────────────────────────────────────────────────
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")

# ── Misc ──────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def validate():
    missing = []
    if not LIVEKIT_URL:
        missing.append("LIVEKIT_URL")
    if not LIVEKIT_API_KEY:
        missing.append("LIVEKIT_API_KEY")
    if not LIVEKIT_API_SECRET:
        missing.append("LIVEKIT_API_SECRET")
    if missing:
        logger.error(f"Missing required config: {', '.join(missing)}")
        return False
    return True
