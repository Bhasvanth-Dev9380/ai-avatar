#!/usr/bin/env python3
"""Download all MuseTalk model weights using huggingface_hub."""

import os
import sys

# Ensure we can import
from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = "/workspace/MuseTalk/models"


def download_file(repo_id, filename, local_dir, subfolder=None):
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest):
        print(f"  already exists: {filename}")
        return True
    print(f"  downloading: {repo_id} / {filename} ...")
    try:
        hf_hub_download(
            repo_id,
            filename,
            local_dir=local_dir,
            subfolder=subfolder,
        )
        print(f"  done: {filename}")
        return True
    except Exception as e:
        print(f"  FAILED: {filename} — {e}")
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── MuseTalk V1.0 weights ──
    print("\n=== MuseTalk V1.0 weights ===")
    v1_dir = os.path.join(MODELS_DIR, "musetalk")
    os.makedirs(v1_dir, exist_ok=True)
    download_file("TMElyralab/MuseTalk", "musetalk.json", v1_dir)
    download_file("TMElyralab/MuseTalk", "pytorch_model.bin", v1_dir)

    # ── MuseTalk V1.5 weights ──
    print("\n=== MuseTalk V1.5 weights ===")
    v15_dir = os.path.join(MODELS_DIR, "musetalkV15")
    os.makedirs(v15_dir, exist_ok=True)
    download_file("TMElyralab/MuseTalk", "musetalk.json", v15_dir)
    download_file("TMElyralab/MuseTalk", "unet.pth", v15_dir)

    # ── Whisper tiny ──
    print("\n=== Whisper tiny ===")
    whisper_dir = os.path.join(MODELS_DIR, "whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    download_file("openai/whisper-tiny", "config.json", whisper_dir)
    download_file("openai/whisper-tiny", "pytorch_model.bin", whisper_dir)
    download_file("openai/whisper-tiny", "preprocessor_config.json", whisper_dir)

    # ── DWPose ──
    print("\n=== DWPose ===")
    dwpose_dir = os.path.join(MODELS_DIR, "dwpose")
    os.makedirs(dwpose_dir, exist_ok=True)
    download_file("yzd-v/DWPose", "dw-ll_ucoco_384.pth", dwpose_dir)

    # ── SyncNet ──
    print("\n=== SyncNet ===")
    syncnet_dir = os.path.join(MODELS_DIR, "syncnet")
    os.makedirs(syncnet_dir, exist_ok=True)
    download_file("ByteDance/LatentSync", "latentsync_syncnet.pt", syncnet_dir)

    # ── Face Parse BiSeNet ──
    print("\n=== Face Parse BiSeNet ===")
    faceparse_dir = os.path.join(MODELS_DIR, "face-parse-bisent")
    os.makedirs(faceparse_dir, exist_ok=True)
    download_file("TMElyralab/MuseTalk", "79999_iter.pth", faceparse_dir)
    download_file("TMElyralab/MuseTalk", "resnet18-5c106cde.pth", faceparse_dir)

    # ── SD VAE ──
    print("\n=== SD VAE (ft-mse) ===")
    vae_dir = os.path.join(MODELS_DIR, "sd-vae-ft-mse")
    os.makedirs(vae_dir, exist_ok=True)
    download_file("stabilityai/sd-vae-ft-mse", "config.json", vae_dir)
    download_file("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", vae_dir)

    # ── List all model files ──
    print("\n=== Model files ===")
    for root, dirs, files in sorted(os.walk(MODELS_DIR)):
        for f in sorted(files):
            if f.endswith((".bin", ".pth", ".pt", ".json", ".onnx", ".safetensors")):
                path = os.path.join(root, f)
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"  {os.path.relpath(path, MODELS_DIR):50s} {size:8.1f} MB")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
