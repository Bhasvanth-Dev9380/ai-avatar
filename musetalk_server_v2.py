"""
MuseTalk Real-Time Server v2
============================
FastAPI server with PRELOADED models and pre-computed avatar data.

Architecture:
  STARTUP (once):
    1. Load VAE, UNet, PE, Whisper → GPU memory (fp16)
    2. Run face detection, landmarks, VAE encoding for avatar image
    3. Pre-compute masks and blending materials
    → All expensive work is done ONCE

  PER REQUEST (fast):
    1. Extract whisper features from audio (~50ms)
    2. Run UNet (1 step per frame, batched) + VAE decode (~33ms/frame)
    3. Blend onto original frame (~5ms/frame)
    4. Encode frames to MP4 in memory → return

Endpoints:
  GET  /health       — readiness check
  GET  /idle         — returns idle avatar frame as JPEG
  POST /synthesize   — WAV audio → lip-synced MP4 (real-time speed)
  POST /synthesize_frames — WAV audio → raw RGBA frames (for LiveKit)

Run:
  cd /workspace/MuseTalk && uvicorn musetalk_server_v2:app --host 0.0.0.0 --port 7860
"""

import asyncio
import copy
import glob
import io
import logging
import math
import os
import pickle
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

# ── MuseTalk imports (must run from /workspace/MuseTalk) ────────────────────
sys.path.insert(0, "/workspace/MuseTalk")

from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

# ── Configuration ─────────────────────────────────────────────────────────────
AVATAR_IMAGE = os.getenv("AVATAR_IMAGE", "/workspace/avatar.jpg")
GPU_ID = int(os.getenv("GPU_ID", "0"))
FPS = int(os.getenv("AVATAR_FPS", "25"))
BATCH_SIZE = int(os.getenv("MUSETALK_BATCH_SIZE", "16"))
VERSION = os.getenv("MUSETALK_VERSION", "v15")
EXTRA_MARGIN = int(os.getenv("EXTRA_MARGIN", "10"))
PARSING_MODE = os.getenv("PARSING_MODE", "jaw")
BBOX_SHIFT = int(os.getenv("BBOX_SHIFT", "0"))
AUDIO_PAD_LEFT = int(os.getenv("AUDIO_PAD_LEFT", "2"))
AUDIO_PAD_RIGHT = int(os.getenv("AUDIO_PAD_RIGHT", "2"))

logger = logging.getLogger("musetalk-v2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Global state ──────────────────────────────────────────────────────────────
device = None
vae = None
unet = None
pe = None
whisper = None
audio_processor = None
weight_dtype = None
timesteps = None
fp = None  # FaceParsing

# Avatar pre-computed data
avatar_ready = False
frame_list_cycle = []
coord_list_cycle = []
input_latent_list_cycle = []
mask_list_cycle = []
mask_coords_list_cycle = []

# Idle frame
_idle_jpeg: bytes | None = None


# ── Avatar preparation ────────────────────────────────────────────────────────

def prepare_avatar(avatar_path: str, cache_dir: str = "/workspace/avatar_cache"):
    """
    Pre-compute face detection, landmarks, VAE latents, masks for the avatar.
    Results are cached to disk for fast reload.
    """
    global frame_list_cycle, coord_list_cycle, input_latent_list_cycle
    global mask_list_cycle, mask_coords_list_cycle, avatar_ready, _idle_jpeg

    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file paths
    latents_path = os.path.join(cache_dir, "latents.pt")
    coords_path = os.path.join(cache_dir, "coords.pkl")
    mask_coords_path = os.path.join(cache_dir, "mask_coords.pkl")
    frames_path = os.path.join(cache_dir, "full_imgs")
    masks_path = os.path.join(cache_dir, "masks")
    info_path = os.path.join(cache_dir, "info.json")
    
    import json

    # Check if cache is valid
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        img_mtime = os.path.getmtime(avatar_path)
        if (info.get("avatar_path") == avatar_path 
            and info.get("mtime") == img_mtime
            and os.path.exists(latents_path)
            and os.path.exists(coords_path)):
            logger.info("Loading avatar data from cache...")
            _load_avatar_cache(cache_dir)
            avatar_ready = True
            logger.info(f"Avatar ready from cache: {len(frame_list_cycle)} frames")
            return

    logger.info("Preparing avatar (first time — will be cached)...")
    t0 = time.time()

    # Clear cache
    if os.path.exists(frames_path):
        shutil.rmtree(frames_path)
    if os.path.exists(masks_path):
        shutil.rmtree(masks_path)
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)

    # For a single image, create a one-frame "video"
    img = cv2.imread(avatar_path)
    if img is None:
        raise RuntimeError(f"Cannot read avatar image: {avatar_path}")
    
    frame_path = os.path.join(frames_path, "00000000.png")
    cv2.imwrite(frame_path, img)
    
    input_img_list = sorted(glob.glob(os.path.join(frames_path, '*.[jpJP][pnPN]*[gG]')))
    
    logger.info("  Extracting landmarks and bounding boxes...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, BBOX_SHIFT)
    
    input_latent_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    
    for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        if VERSION == "v15":
            y2 = y2 + EXTRA_MARGIN
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(resized_crop_frame)
        input_latent_list.append(latents)
    
    # Create cycle (forward + reverse for smooth looping)
    _frame_list_cycle = frame_list + frame_list[::-1]
    _coord_list_cycle = coord_list + coord_list[::-1]
    _input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    logger.info("  Computing face masks...")
    _mask_coords_list_cycle = []
    _mask_list_cycle = []
    
    for i, frame in enumerate(_frame_list_cycle):
        cv2.imwrite(os.path.join(frames_path, f"{str(i).zfill(8)}.png"), frame)
        x1, y1, x2, y2 = _coord_list_cycle[i]
        mode = PARSING_MODE if VERSION == "v15" else "raw"
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)
        cv2.imwrite(os.path.join(masks_path, f"{str(i).zfill(8)}.png"), mask)
        _mask_coords_list_cycle.append(crop_box)
        _mask_list_cycle.append(mask)
    
    # Save cache
    torch.save(_input_latent_list_cycle, latents_path)
    with open(coords_path, 'wb') as f:
        pickle.dump(_coord_list_cycle, f)
    with open(mask_coords_path, 'wb') as f:
        pickle.dump(_mask_coords_list_cycle, f)
    
    img_mtime = os.path.getmtime(avatar_path)
    with open(info_path, "w") as f:
        json.dump({"avatar_path": avatar_path, "mtime": img_mtime, "version": VERSION}, f)
    
    # Set globals
    frame_list_cycle = _frame_list_cycle
    coord_list_cycle = _coord_list_cycle
    input_latent_list_cycle = _input_latent_list_cycle
    mask_list_cycle = _mask_list_cycle
    mask_coords_list_cycle = _mask_coords_list_cycle
    
    # Create idle frame JPEG
    idle_frame = frame_list_cycle[0]
    _, jpeg_buf = cv2.imencode('.jpg', idle_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _idle_jpeg = jpeg_buf.tobytes()
    
    avatar_ready = True
    elapsed = time.time() - t0
    logger.info(f"Avatar preparation complete: {len(frame_list_cycle)} frames in {elapsed:.1f}s (cached for next time)")


def _load_avatar_cache(cache_dir: str):
    """Load pre-computed avatar data from cache."""
    global frame_list_cycle, coord_list_cycle, input_latent_list_cycle
    global mask_list_cycle, mask_coords_list_cycle, _idle_jpeg
    
    latents_path = os.path.join(cache_dir, "latents.pt")
    coords_path = os.path.join(cache_dir, "coords.pkl")
    mask_coords_path = os.path.join(cache_dir, "mask_coords.pkl")
    frames_path = os.path.join(cache_dir, "full_imgs")
    masks_path = os.path.join(cache_dir, "masks")
    
    input_latent_list_cycle = torch.load(latents_path, weights_only=False)
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    
    # Read frames
    input_img_list = sorted(glob.glob(os.path.join(frames_path, '*.[jpJP][pnPN]*[gG]')))
    frame_list_cycle = read_imgs(input_img_list)
    
    # Read masks
    input_mask_list = sorted(glob.glob(os.path.join(masks_path, '*.[jpJP][pnPN]*[gG]')))
    mask_list_cycle = read_imgs(input_mask_list)
    
    # Create idle frame JPEG
    if frame_list_cycle:
        _, jpeg_buf = cv2.imencode('.jpg', frame_list_cycle[0], [cv2.IMWRITE_JPEG_QUALITY, 90])
        _idle_jpeg = jpeg_buf.tobytes()


# ── Inference (the fast path) ─────────────────────────────────────────────────

@torch.no_grad()
def synthesize_frames(audio_path: str) -> list[np.ndarray]:
    """
    Given an audio WAV file, run the FAST inference path:
    1. Extract whisper features
    2. UNet inference (single step, batched)
    3. VAE decode
    4. Blend onto original frames
    
    Returns list of BGR numpy arrays (full resolution).
    """
    t_total = time.time()
    
    # Step 1: Audio features
    t0 = time.time()
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(
        audio_path, weight_dtype=weight_dtype
    )
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=FPS,
        audio_padding_length_left=AUDIO_PAD_LEFT,
        audio_padding_length_right=AUDIO_PAD_RIGHT,
    )
    logger.info(f"  Audio features: {(time.time() - t0) * 1000:.0f}ms ({len(whisper_chunks)} chunks)")
    
    # Step 2+3: UNet + VAE decode (batched)
    t0 = time.time()
    video_num = len(whisper_chunks)
    gen = datagen(whisper_chunks, input_latent_list_cycle, BATCH_SIZE)
    
    res_frames = []
    for whisper_batch, latent_batch in gen:
        audio_feature_batch = pe(whisper_batch.to(device))
        latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
        
        pred_latents = unet.model(
            latent_batch,
            timesteps,
            encoder_hidden_states=audio_feature_batch,
        ).sample
        pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
        recon = vae.decode_latents(pred_latents)
        for frame in recon:
            res_frames.append(frame)
    
    logger.info(f"  UNet+VAE: {(time.time() - t0) * 1000:.0f}ms ({len(res_frames)} frames)")
    
    # Step 4: Blend onto original frames
    t0 = time.time()
    output_frames = []
    for idx, res_frame in enumerate(res_frames):
        bbox = coord_list_cycle[idx % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[idx % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:
            continue
        mask = mask_list_cycle[idx % len(mask_list_cycle)]
        mask_crop_box = mask_coords_list_cycle[idx % len(mask_coords_list_cycle)]
        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        output_frames.append(combine_frame)
    
    logger.info(f"  Blending: {(time.time() - t0) * 1000:.0f}ms")
    logger.info(f"  TOTAL: {(time.time() - t_total) * 1000:.0f}ms for {len(output_frames)} frames ({len(output_frames)/FPS:.2f}s video)")
    
    return output_frames


def frames_to_mp4(frames: list[np.ndarray], audio_path: str, fps: int = FPS) -> bytes:
    """Encode frames + audio to MP4 in memory using ffmpeg pipe."""
    if not frames:
        return b""
    
    h, w = frames[0].shape[:2]
    
    # Use ffmpeg with pipe input for video and file input for audio
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        # Video input from pipe
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "pipe:0",
        # Audio input
        "-i", audio_path,
        # Output
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", "23",
        "-vf", "format=yuv420p",
        "-shortest",
        "-movflags", "+faststart+frag_keyframe+empty_moov",
        "-f", "mp4",
        "pipe:1",
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Write frames to stdin
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    
    stdout, stderr = proc.communicate()
    
    if proc.returncode != 0:
        logger.error(f"ffmpeg error: {stderr.decode(errors='replace')[-500:]}")
        raise RuntimeError("ffmpeg encoding failed")
    
    return stdout


def frames_to_rgba_bytes(frames: list[np.ndarray], width: int = 256, height: int = 256) -> bytes:
    """Convert frames to packed RGBA bytes for direct LiveKit consumption."""
    all_bytes = b""
    for frame in frames:
        resized = cv2.resize(frame, (width, height))
        rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        all_bytes += rgba.tobytes()
    return all_bytes


# ── FastAPI App ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and prepare avatar at startup."""
    global device, vae, unet, pe, whisper, audio_processor
    global weight_dtype, timesteps, fp
    
    logger.info("=" * 60)
    logger.info("MuseTalk Real-Time Server v2 — Starting up")
    logger.info("=" * 60)
    
    # Set device
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load models
    t0 = time.time()
    logger.info("Loading VAE, UNet, PE models...")
    
    if VERSION == "v15":
        unet_config = os.path.join("models", "musetalkV15", "musetalk.json")
        unet_model_path = os.path.join("models", "musetalkV15", "unet.pth")
    else:
        unet_config = os.path.join("models", "musetalk", "musetalk.json")
        unet_model_path = os.path.join("models", "musetalk", "pytorch_model.bin")
    
    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type="sd-vae",
        unet_config=unet_config,
        device=device,
    )
    
    # Move to fp16 on GPU
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    
    timesteps = torch.tensor([0], device=device)
    weight_dtype = unet.model.dtype
    
    logger.info(f"Models loaded in {time.time() - t0:.1f}s")
    
    # Load Whisper
    t0 = time.time()
    whisper_dir = os.path.join("models", "whisper")
    logger.info(f"Loading Whisper from {whisper_dir}...")
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    logger.info(f"Whisper loaded in {time.time() - t0:.1f}s")
    
    # Initialize face parser
    fp = FaceParsing()
    
    # Prepare avatar
    if os.path.exists(AVATAR_IMAGE):
        logger.info(f"Preparing avatar from {AVATAR_IMAGE}...")
        prepare_avatar(AVATAR_IMAGE)
    else:
        logger.warning(f"Avatar image not found: {AVATAR_IMAGE}")

    logger.info("=" * 60)
    logger.info("SERVER READY — Models in GPU, avatar pre-computed")
    logger.info("=" * 60)
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")


app = FastAPI(title="MuseTalk Real-Time Server v2", lifespan=lifespan)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ready" if avatar_ready else "loading",
        "ready": avatar_ready,
        "version": "v2-realtime",
        "avatar_frames": len(frame_list_cycle),
        "gpu": f"cuda:{GPU_ID}",
    }


@app.get("/idle")
def get_idle():
    """Return the avatar idle frame as JPEG."""
    if not _idle_jpeg:
        raise HTTPException(503, "Avatar not ready yet")
    return Response(content=_idle_jpeg, media_type="image/jpeg")


@app.post("/synthesize")
async def synthesize(audio: UploadFile = File(...)):
    """
    Generate a lip-synced avatar video from WAV audio.
    
    Input:  multipart/form-data field 'audio' (WAV, 16kHz recommended)
    Output: video/mp4
    """
    if not avatar_ready:
        raise HTTPException(503, "Avatar not ready yet — still loading models")
    
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")
    
    logger.info(f"/synthesize — {len(audio_bytes):,} bytes audio received")
    t0 = time.time()
    
    # Write audio to temp file (whisper needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name
    
    try:
        # Run inference in thread pool to not block event loop
        frames = await asyncio.to_thread(synthesize_frames, audio_path)
        
        if not frames:
            raise HTTPException(500, "No frames generated")
        
        # Encode to MP4
        video_bytes = await asyncio.to_thread(frames_to_mp4, frames, audio_path)
        
        elapsed = time.time() - t0
        audio_duration = len(audio_bytes) / (16000 * 2)  # rough estimate for 16kHz 16-bit mono
        rtf = elapsed / max(audio_duration, 0.1)
        logger.info(f"/synthesize OK — {len(frames)} frames, {len(video_bytes):,} bytes, "
                     f"{elapsed:.2f}s (RTF: {rtf:.2f}x)")
        
        return Response(content=video_bytes, media_type="video/mp4")
    
    finally:
        os.unlink(audio_path)


@app.post("/synthesize_frames")
async def synthesize_frames_endpoint(audio: UploadFile = File(...)):
    """
    Generate lip-synced RGBA frames from WAV audio.
    Returns raw RGBA bytes (256x256 per frame) for direct LiveKit usage.
    
    Response headers include:
      X-Frame-Count: number of frames
      X-Frame-Width: 256
      X-Frame-Height: 256
      X-FPS: 25
    """
    if not avatar_ready:
        raise HTTPException(503, "Avatar not ready yet")
    
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")
    
    logger.info(f"/synthesize_frames — {len(audio_bytes):,} bytes audio")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name
    
    try:
        frames = await asyncio.to_thread(synthesize_frames, audio_path)
        
        if not frames:
            raise HTTPException(500, "No frames generated")
        
        rgba_bytes = await asyncio.to_thread(frames_to_rgba_bytes, frames)
        
        return Response(
            content=rgba_bytes,
            media_type="application/octet-stream",
            headers={
                "X-Frame-Count": str(len(frames)),
                "X-Frame-Width": "256",
                "X-Frame-Height": "256",
                "X-FPS": str(FPS),
            },
        )
    finally:
        os.unlink(audio_path)


@app.post("/prepare_avatar")
async def prepare_avatar_endpoint(image: UploadFile = File(...)):
    """
    Upload a new avatar image and re-run preparation.
    """
    global avatar_ready
    avatar_ready = False
    
    img_bytes = await image.read()
    
    # Save the new avatar image
    with open(AVATAR_IMAGE, "wb") as f:
        f.write(img_bytes)
    
    # Clear cache and re-prepare
    cache_dir = "/workspace/avatar_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    try:
        await asyncio.to_thread(prepare_avatar, AVATAR_IMAGE, cache_dir)
        return {"status": "ok", "frames": len(frame_list_cycle)}
    except Exception as e:
        raise HTTPException(500, f"Avatar preparation failed: {e}")
