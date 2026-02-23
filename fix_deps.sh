#!/bin/bash
set -e
VENV=/workspace/.venv

echo "=== Fixing Python 3.12 compatibility ==="

# Upgrade pip and setuptools
$VENV/bin/pip install --upgrade pip setuptools wheel -q
echo "pip/setuptools upgraded"

# Pre-install numpy compatible with 3.12
$VENV/bin/pip install "numpy>=1.26,<2" -q
echo "numpy installed"

# Install MuseTalk requirements
echo "Installing MuseTalk deps (this may take a few minutes)..."
$VENV/bin/pip install -r /workspace/MuseTalk/requirements.txt -q 2>&1 | tail -10
echo "MuseTalk deps done"

# mmcv
echo "Installing mmcv..."
$VENV/bin/pip install mmcv -q 2>&1 | tail -5 || echo "mmcv install had issues, continuing..."

# huggingface_hub for model downloads
$VENV/bin/pip install huggingface_hub -q

# Download MuseTalk models
echo "=== Downloading MuseTalk models ==="
MODELS=/workspace/MuseTalk/models
mkdir -p "$MODELS/musetalk" "$MODELS/dwpose" "$MODELS/sd-vae-ft-mse" "$MODELS/whisper" "$MODELS/face-parse-bisent"

$VENV/bin/python << 'PYEOF'
from huggingface_hub import hf_hub_download
import os

MODELS = "/workspace/MuseTalk/models"

downloads = [
    ("TMElyralab/MuseTalk", "models/musetalk/pytorch_model.bin"),
    ("TMElyralab/MuseTalk", "models/musetalk/musetalk.json"),
    ("TMElyralab/MuseTalk", "models/dwpose/dw-ll_ucoco_384.onnx"),
    ("TMElyralab/MuseTalk", "models/face-parse-bisent/79999_iter.pth"),
    ("TMElyralab/MuseTalk", "models/face-parse-bisent/resnet18-5c106cde.pth"),
]

for repo, filename in downloads:
    dest = os.path.join("/workspace/MuseTalk", filename)
    if os.path.exists(dest):
        print(f"  exists: {os.path.basename(filename)}")
        continue
    print(f"  downloading: {filename}...")
    try:
        hf_hub_download(repo, filename, local_dir="/workspace/MuseTalk")
        print(f"  done")
    except Exception as e:
        print(f"  WARN: {e}")

# VAE
for fn in ["diffusion_pytorch_model.bin", "config.json"]:
    dest = f"{MODELS}/sd-vae-ft-mse/{fn}"
    if os.path.exists(dest):
        print(f"  exists: {fn}")
        continue
    print(f"  downloading: sd-vae-ft-mse/{fn}...")
    try:
        hf_hub_download("stabilityai/sd-vae-ft-mse", fn, local_dir=f"{MODELS}/sd-vae-ft-mse")
    except Exception as e:
        print(f"  WARN: {e}")

print("Model downloads complete")
PYEOF

echo ""
echo "=== Setup complete ==="
echo "Run: bash /workspace/ai-avatar/start.sh"
