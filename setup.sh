#!/bin/bash
# ============================================================
#  AI Avatar — RunPod Setup Script
#
#  Run once after SSHing into a fresh RunPod GPU pod:
#    bash /workspace/ai-avatar/setup.sh
#
#  What this does:
#   1. System packages (ffmpeg, git-lfs, build tools)
#   2. Python venv with PyTorch + all deps
#   3. Clone MuseTalk and download its models
#   4. Install VibeVoice TTS (from persistent storage)
#   5. Install Ollama (for local LLM)
# ============================================================
set -e

WORKSPACE=/workspace
AVATAR_DIR="$WORKSPACE/ai-avatar"
VENV="$WORKSPACE/.venv"
MUSETALK_DIR="$WORKSPACE/MuseTalk"

echo "============================================"
echo "  AI Avatar Setup"
echo "============================================"

# ── 1. System packages ──────────────────────────────────────
echo ""
echo "=== Step 1: System packages ==="
apt-get update -qq
apt-get install -y -qq git git-lfs ffmpeg libsm6 libxext6 libgl1 build-essential > /dev/null 2>&1
git lfs install > /dev/null 2>&1
echo "✓ System packages installed"

# ── 2. Python venv ─────────────────────────────────────────
echo ""
echo "=== Step 2: Python virtual environment ==="
if [ ! -f "$VENV/bin/python" ]; then
    python3.10 -m venv "$VENV" 2>/dev/null || python3 -m venv "$VENV"
    echo "  Created new venv at $VENV"
fi
"$VENV/bin/pip" install --upgrade pip -q

echo "  Installing PyTorch (CUDA 12.1)..."
"$VENV/bin/pip" install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 -q

echo "  Installing avatar agent dependencies..."
"$VENV/bin/pip" install -r "$AVATAR_DIR/requirements.txt" -q

echo "✓ Python environment ready"

# ── 3. MuseTalk ─────────────────────────────────────────────
echo ""
echo "=== Step 3: MuseTalk ==="

if [ -d "$MUSETALK_DIR" ]; then
    echo "  MuseTalk already cloned, pulling latest..."
    git -C "$MUSETALK_DIR" pull -q 2>/dev/null || true
else
    echo "  Cloning MuseTalk..."
    git clone https://github.com/TMElyralab/MuseTalk "$MUSETALK_DIR" -q
fi

echo "  Installing MuseTalk dependencies..."
"$VENV/bin/pip" install -r "$MUSETALK_DIR/requirements.txt" -q

# mmcv for MuseTalk (requires special index)
echo "  Installing mmcv..."
"$VENV/bin/pip" install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html -q 2>/dev/null \
    || "$VENV/bin/pip" install mmcv -q

echo "✓ MuseTalk installed"

# ── 4. MuseTalk model weights ───────────────────────────────
echo ""
echo "=== Step 4: Download MuseTalk model weights ==="

MODELS_DIR="$MUSETALK_DIR/models"
mkdir -p "$MODELS_DIR"

download_model() {
    local repo="$1"
    local filename="$2"
    local dest="$3"
    if [ -f "$dest" ]; then
        echo "  ✓ $filename already exists"
        return
    fi
    echo "  Downloading $filename..."
    mkdir -p "$(dirname "$dest")"
    "$VENV/bin/python" -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$repo', '$filename', local_dir='$(dirname "$dest")')
print('  done')
" 2>/dev/null || echo "  ⚠ Download failed for $filename — you may need to download it manually"
}

"$VENV/bin/pip" install huggingface_hub -q

# MuseTalk main weights
download_model "TMElyralab/MuseTalk" \
    "musetalk/pytorch_model.bin" \
    "$MODELS_DIR/musetalk/pytorch_model.bin"
download_model "TMElyralab/MuseTalk" \
    "musetalk/musetalk.json" \
    "$MODELS_DIR/musetalk/musetalk.json"

# DWPose
download_model "TMElyralab/MuseTalk" \
    "dwpose/dw-ll_ucoco_384.onnx" \
    "$MODELS_DIR/dwpose/dw-ll_ucoco_384.onnx"

# VAE
download_model "stabilityai/sd-vae-ft-mse" \
    "diffusion_pytorch_model.bin" \
    "$MODELS_DIR/sd-vae-ft-mse/diffusion_pytorch_model.bin"
download_model "stabilityai/sd-vae-ft-mse" \
    "config.json" \
    "$MODELS_DIR/sd-vae-ft-mse/config.json"

# Whisper tiny (for MuseTalk audio features)
download_model "openai/whisper-tiny" \
    "pytorch_model.bin" \
    "$MODELS_DIR/whisper/tiny.pt" 2>/dev/null \
    || "$VENV/bin/python" -c "import whisper; whisper.load_model('tiny', download_root='$MODELS_DIR/whisper')" 2>/dev/null || true

# face-parse-bisent
download_model "TMElyralab/MuseTalk" \
    "face-parse-bisent/79999_iter.pth" \
    "$MODELS_DIR/face-parse-bisent/79999_iter.pth"
download_model "TMElyralab/MuseTalk" \
    "face-parse-bisent/resnet18-5c106cde.pth" \
    "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"

echo "✓ MuseTalk models ready"

# ── 5. VibeVoice TTS ────────────────────────────────────────
echo ""
echo "=== Step 5: VibeVoice TTS ==="

VIBEVOICE_DIR="/mnt/persistent/VibeVoice"
if [ -d "$VIBEVOICE_DIR" ]; then
    echo "  Installing VibeVoice from persistent storage..."
    "$VENV/bin/pip" install -e "$VIBEVOICE_DIR" -q \
        && echo "✓ VibeVoice installed" \
        || echo "⚠ VibeVoice install failed — check $VIBEVOICE_DIR"
else
    echo "⚠ VibeVoice not found at $VIBEVOICE_DIR"
    echo "  Copy your VibeVoice package to $VIBEVOICE_DIR before running start.sh"
fi

# ── 6. Ollama ───────────────────────────────────────────────
echo ""
echo "=== Step 6: Ollama (local LLM) ==="
if ! command -v ollama &>/dev/null; then
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama already installed"
fi

# ── Done ────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Add your avatar portrait:"
echo "     cp /path/to/your/photo.jpg /workspace/avatar.jpg"
echo ""
echo "  2. Create .env from example:"
echo "     cp $AVATAR_DIR/.env.example $AVATAR_DIR/.env"
echo "     nano $AVATAR_DIR/.env   # fill in LiveKit + API keys"
echo ""
echo "  3. Start everything:"
echo "     bash $AVATAR_DIR/start.sh"
echo "============================================"
