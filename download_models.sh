#!/bin/bash
set -e
cd /workspace/MuseTalk
export PATH="/workspace/.venv/bin:$PATH"

pip install -U "huggingface_hub[cli]" gdown -q

echo "=== Downloading MuseTalk V1.0 weights ==="
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir models \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

echo "=== Downloading MuseTalk V1.5 weights ==="
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir models \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

echo "=== Downloading Whisper ==="
huggingface-cli download openai/whisper-tiny \
  --local-dir models/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

echo "=== Downloading DWPose ==="
huggingface-cli download yzd-v/DWPose \
  --local-dir models/dwpose \
  --include "dw-ll_ucoco_384.pth"

echo "=== Downloading SyncNet ==="
huggingface-cli download ByteDance/LatentSync \
  --local-dir models/syncnet \
  --include "latentsync_syncnet.pt"

echo "=== Downloading Face Parse ==="
if [ ! -f models/face-parse-bisent/79999_iter.pth ]; then
    gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
fi

echo ""
echo "=== Model files ==="
find models -type f \( -name "*.bin" -o -name "*.pth" -o -name "*.json" -o -name "*.pt" \) | sort
echo "=== ALL DONE ==="
