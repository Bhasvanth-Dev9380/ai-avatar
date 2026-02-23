#!/bin/bash
set -e
VENV=/workspace/.venv
PIP="$VENV/bin/pip"

echo "=== Step 1: Fix MuseTalk deps for Python 3.12 ==="

# Upgrade build tools to Python 3.12 compatible versions
$PIP install --upgrade pip "setuptools>=75" wheel -q

# Install numpy that works with 3.12 FIRST
$PIP install "numpy>=1.26,<2" -q

# Install MuseTalk deps one-by-one, skipping the broken pinned versions
# Skip: numpy==1.23.5 (incompatible), tensorflow (not needed for inference)
$PIP install "diffusers==0.30.2" "accelerate==0.28.0" -q
$PIP install "opencv-python==4.9.0.80" "soundfile==0.12.1" -q 2>/dev/null || $PIP install opencv-python-headless soundfile -q
$PIP install "transformers==4.39.2" "huggingface_hub==0.30.2" -q
$PIP install "librosa==0.11.0" "einops==0.8.1" -q
$PIP install gdown requests "imageio[ffmpeg]" -q
$PIP install omegaconf ffmpeg-python moviepy -q
$PIP install gradio -q 2>/dev/null || echo "gradio skipped (not needed for server)"

# mmlab deps (needed for dwpose)
$PIP install openmim -q 2>/dev/null || true
$PIP install mmengine -q 2>/dev/null || true
$PIP install mmdet -q 2>/dev/null || true
$PIP install mmpose -q 2>/dev/null || true

echo "MuseTalk deps installed"

echo "=== Step 2: Download MuseTalk model weights ==="
cd /workspace/MuseTalk

# Use their official download script
if [ -f "download_weights.sh" ]; then
    echo "Running MuseTalk download_weights.sh..."
    bash download_weights.sh 2>&1 | tail -20
else
    echo "download_weights.sh not found, downloading manually..."
    $VENV/bin/python << 'PYEOF'
from huggingface_hub import snapshot_download, hf_hub_download
import os

MODELS = "/workspace/MuseTalk/models"

# MuseTalk weights from their HF repo (top-level, not under models/)
print("Downloading MuseTalk weights...")
try:
    snapshot_download(
        "TMElyralab/MuseTalk",
        local_dir="/workspace/MuseTalk/models/musetalk_hf",
        allow_patterns=["*.bin", "*.json", "*.pth"],
        ignore_patterns=["*.md", "*.txt"],
    )
    print("  done")
except Exception as e:
    print(f"  snapshot failed: {e}")
    # Try individual files
    for fn in ["musetalk.json", "pytorch_model.bin"]:
        dest = f"{MODELS}/musetalk/{fn}"
        if os.path.exists(dest):
            print(f"  exists: {fn}")
            continue
        try:
            hf_hub_download("TMElyralab/MuseTalk", fn, local_dir=f"{MODELS}/musetalk")
            print(f"  downloaded: {fn}")
        except Exception as e2:
            print(f"  {fn}: {e2}")

# Whisper tiny
print("Downloading whisper-tiny...")
try:
    dest = f"{MODELS}/whisper"
    os.makedirs(dest, exist_ok=True)
    hf_hub_download("openai/whisper-tiny", "model.safetensors", local_dir=dest)
    print("  done")
except Exception as e:
    print(f"  whisper: {e}")

# DWPose
print("Downloading DWPose...")
try:
    dest = f"{MODELS}/dwpose"
    os.makedirs(dest, exist_ok=True)
    hf_hub_download("yzd-v/DWPose", "dw-ll_ucoco_384.pth", local_dir=dest)
    print("  done")
except Exception as e:
    print(f"  dwpose: {e}")

print("Model downloads complete")
PYEOF
fi

echo ""
echo "=== Checking model files ==="
find /workspace/MuseTalk/models -name "*.bin" -o -name "*.pth" -o -name "*.pt" -o -name "*.json" -o -name "*.onnx" -o -name "*.safetensors" 2>/dev/null | sort

echo ""
echo "=== Setup complete ==="
