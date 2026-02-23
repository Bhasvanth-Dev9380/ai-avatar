#!/bin/bash
# ============================================================
#  AI Avatar — Start All Services
#
#  Run on the RunPod pod after setup.sh:
#    bash /workspace/ai-avatar/start.sh
#
#  Starts:
#   1. Ollama (LLM server, port 11434)
#   2. VibeVoice TTS server (port 9000)
#   3. MuseTalk avatar server (port 7860)
#   4. LiveKit avatar agent (connects to LiveKit Cloud)
# ============================================================

WORKSPACE=/workspace
AVATAR_DIR="$WORKSPACE/ai-avatar"
VENV="$WORKSPACE/.venv"
PYTHON="$VENV/bin/python"
LOG_DIR="$WORKSPACE/logs"

mkdir -p "$LOG_DIR"

# ── Guard: .env must exist ──────────────────────────────────
if [ ! -f "$AVATAR_DIR/.env" ]; then
    echo "ERROR: $AVATAR_DIR/.env not found."
    echo "  cp $AVATAR_DIR/.env.example $AVATAR_DIR/.env"
    echo "  nano $AVATAR_DIR/.env   # fill in your credentials"
    exit 1
fi

# ── Load env so we can print useful info ───────────────────
set -a; source "$AVATAR_DIR/.env"; set +a

echo "============================================"
echo "  AI Avatar — Starting Services"
echo "============================================"
echo "  LiveKit : $LIVEKIT_URL"
echo "  LLM     : $SELFHOSTED_LLM_MODEL @ $SELFHOSTED_LLM_URL"
echo "  TTS     : $VIBEVOICE_VOICE @ $VIBEVOICE_TTS_URL"
echo "  Avatar  : $AVATAR_IMAGE"
echo "============================================"
echo ""

# ── 1. Ollama ──────────────────────────────────────────────
echo "[1/4] Starting Ollama..."
if ! pgrep -x ollama > /dev/null; then
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    sleep 3

    # Pull model if not already present
    MODEL_NAME="${SELFHOSTED_LLM_MODEL:-qwen2.5:7b}"
    if ! ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
        echo "  Pulling $MODEL_NAME (this may take a while)..."
        ollama pull "$MODEL_NAME" 2>&1 | tail -5
    fi
    echo "  Ollama started (PID $OLLAMA_PID)"
else
    echo "  Ollama already running"
fi

# ── 2. VibeVoice TTS ───────────────────────────────────────
echo "[2/4] Starting VibeVoice TTS server..."
VIBEVOICE_DIR="/mnt/persistent/VibeVoice"
TTS_PORT=9000

if lsof -ti:"$TTS_PORT" > /dev/null 2>&1; then
    echo "  Port $TTS_PORT already in use — assuming TTS is running"
else
    if [ -f "$VIBEVOICE_DIR/server.py" ]; then
        nohup "$PYTHON" "$VIBEVOICE_DIR/server.py" \
            --port "$TTS_PORT" \
            > "$LOG_DIR/vibevoice.log" 2>&1 &
        echo "  VibeVoice started (log: $LOG_DIR/vibevoice.log)"
    elif [ -f "$VIBEVOICE_DIR/app.py" ]; then
        nohup "$PYTHON" "$VIBEVOICE_DIR/app.py" \
            --port "$TTS_PORT" \
            > "$LOG_DIR/vibevoice.log" 2>&1 &
        echo "  VibeVoice started via app.py (log: $LOG_DIR/vibevoice.log)"
    else
        echo "  ⚠ VibeVoice server entry point not found in $VIBEVOICE_DIR"
        echo "    Adjust VIBEVOICE_TTS_URL in .env if TTS is running elsewhere."
    fi
fi

# ── 3. MuseTalk server ─────────────────────────────────────
echo "[3/4] Starting MuseTalk avatar server (port 7860)..."
MUSETALK_PORT=7860

if lsof -ti:"$MUSETALK_PORT" > /dev/null 2>&1; then
    echo "  Port $MUSETALK_PORT already in use — assuming MuseTalk is running"
else
    MUSETALK_SERVER_URL="http://localhost:$MUSETALK_PORT" \
    AVATAR_IMAGE="${AVATAR_IMAGE:-$WORKSPACE/avatar.jpg}" \
    MUSETALK_DIR="${WORKSPACE}/MuseTalk" \
    nohup "$PYTHON" -m uvicorn musetalk_server:app \
        --app-dir "$AVATAR_DIR" \
        --host 0.0.0.0 \
        --port "$MUSETALK_PORT" \
        > "$LOG_DIR/musetalk.log" 2>&1 &
    echo "  MuseTalk server started (log: $LOG_DIR/musetalk.log)"
    echo "  Startup takes ~60s while pre-rendering idle animation..."
fi

# ── 4. Wait for MuseTalk to be ready ───────────────────────
echo ""
echo "  Waiting for MuseTalk server to become ready..."
MAX_WAIT=180  # 3 minutes
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -s "http://localhost:$MUSETALK_PORT/health" 2>/dev/null \
            | grep -q '"ready": true'; then
        echo "  MuseTalk ready after ${i}s"
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "  ⚠ MuseTalk not ready after ${MAX_WAIT}s — agent will start anyway"
        echo "    Check $LOG_DIR/musetalk.log for errors"
    fi
    sleep 1
done

# ── 5. LiveKit Avatar Agent ────────────────────────────────
echo ""
echo "[4/4] Starting LiveKit avatar agent..."
cd "$AVATAR_DIR"
"$PYTHON" avatar_agent.py start
