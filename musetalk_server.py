"""
MuseTalk Server
FastAPI wrapper around MuseTalk lip-sync inference.

Endpoints:
  GET  /health       — readiness check
  GET  /idle         — returns pre-rendered idle-loop MP4
  POST /synthesize   — accepts a WAV audio upload, returns lip-synced MP4

Run with:
  uvicorn musetalk_server:app --host 0.0.0.0 --port 7860
"""

import asyncio
import glob
import logging
import os
import sys
import tempfile

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

logger = logging.getLogger("musetalk-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="MuseTalk Avatar Server")

MUSETALK_DIR = os.getenv("MUSETALK_DIR", "/workspace/MuseTalk")
AVATAR_IMAGE = os.getenv("AVATAR_IMAGE", "/workspace/avatar.jpg")
FPS = int(os.getenv("AVATAR_FPS", "25"))
BATCH_SIZE = int(os.getenv("MUSETALK_BATCH_SIZE", "8"))
USE_FLOAT16 = os.getenv("MUSETALK_FLOAT16", "true").lower() == "true"
VERSION = os.getenv("MUSETALK_VERSION", "v15")

# Cached idle loop video bytes (generated at startup)
_idle_video_bytes: bytes | None = None
_ready = False


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _ffmpeg_silence(duration: float, out_path: str) -> None:
    """Generate a silent WAV file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r=16000:cl=mono",
        "-t", str(duration),
        "-ar", "16000", "-ac", "1",
        out_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


def _write_inference_config(config_path: str, audio_path: str, video_path: str) -> None:
    """Write a MuseTalk inference YAML config file."""
    with open(config_path, "w") as f:
        f.write(f'task_0:\n')
        f.write(f' video_path: "{video_path}"\n')
        f.write(f' audio_path: "{audio_path}"\n')


async def _run_musetalk(audio_path: str, result_dir: str, vid_name: str = "result.mp4") -> str:
    """
    Run MuseTalk inference and return path to the output video.
    Raises RuntimeError if inference fails.
    """
    # Write inference config YAML
    config_path = os.path.join(result_dir, "inference_config.yaml")
    _write_inference_config(config_path, audio_path, AVATAR_IMAGE)

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", config_path,
        "--result_dir", result_dir,
        "--output_vid_name", vid_name,
        "--fps", str(FPS),
        "--batch_size", str(BATCH_SIZE),
        "--version", VERSION,
    ]
    if USE_FLOAT16:
        cmd.append("--use_float16")

    logger.info(f"Running MuseTalk: {' '.join(cmd[-8:])}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=MUSETALK_DIR,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode(errors="replace")[-800:]
        out = stdout.decode(errors="replace")[-400:]
        logger.error(f"MuseTalk stdout:\n{out}")
        logger.error(f"MuseTalk stderr:\n{err}")
        raise RuntimeError(f"MuseTalk exited {proc.returncode}: {err}")

    # MuseTalk writes output to result_dir/<version>/<vid_name>
    # Try several possible locations
    candidates = [
        os.path.join(result_dir, vid_name),
        os.path.join(result_dir, VERSION, vid_name),
    ]
    # Also search for any .mp4 in result_dir recursively
    for mp4 in glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True):
        if mp4 not in candidates:
            candidates.append(mp4)

    for candidate in candidates:
        if os.path.exists(candidate):
            logger.info(f"Found output video: {candidate}")
            return candidate

    # Log what files exist for debugging
    for root, dirs, files in os.walk(result_dir):
        for f in files:
            logger.info(f"  result file: {os.path.join(root, f)}")

    raise RuntimeError("MuseTalk completed but output video not found")


# ── Startup: pre-render idle loop ─────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    global _idle_video_bytes, _ready

    if not os.path.exists(AVATAR_IMAGE):
        logger.warning(
            f"Avatar image not found: {AVATAR_IMAGE}\n"
            "Upload your portrait photo before sending synthesis requests."
        )
        _ready = True  # Mark ready anyway so agent can start
        return

    if not os.path.exists(MUSETALK_DIR):
        logger.error(f"MuseTalk directory not found: {MUSETALK_DIR}. Run setup.sh first.")
        return

    logger.info("Pre-rendering idle animation loop (3 s silence)...")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            silence = os.path.join(tmp, "silence.wav")
            await _ffmpeg_silence(3.0, silence)
            video_path = await _run_musetalk(silence, tmp, "idle.mp4")
            async with aiofiles.open(video_path, "rb") as fh:
                _idle_video_bytes = await fh.read()

        _ready = True
        logger.info(f"MuseTalk ready. Idle video: {len(_idle_video_bytes):,} bytes")
    except Exception as exc:
        logger.error(f"Startup pre-render failed: {exc}")
        logger.warning("Server will still accept /synthesize requests.")
        _ready = True  # Allow requests anyway


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ready" if _ready else "starting", "ready": _ready}


@app.get("/idle")
def get_idle() -> Response:
    """Return the pre-rendered idle-loop video (loopable MP4)."""
    if not _idle_video_bytes:
        raise HTTPException(status_code=503, detail="Idle video not ready yet")
    return Response(content=_idle_video_bytes, media_type="video/mp4")


@app.post("/synthesize")
async def synthesize(audio: UploadFile = File(...)) -> Response:
    """
    Generate a lip-synced avatar video from a WAV audio upload.

    Input : multipart/form-data  field 'audio'  (WAV, 16kHz mono recommended)
    Output: video/mp4  (256x256, 25 FPS, duration matches input audio)
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    logger.info(f"/synthesize — {len(audio_bytes):,} bytes audio received")

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "input.wav")
        async with aiofiles.open(audio_path, "wb") as fh:
            await fh.write(audio_bytes)

        try:
            video_path = await _run_musetalk(audio_path, tmp)
        except RuntimeError as exc:
            logger.error(f"/synthesize failed: {exc}")
            raise HTTPException(500, str(exc))

        async with aiofiles.open(video_path, "rb") as fh:
            video_bytes = await fh.read()

    logger.info(f"/synthesize OK — {len(video_bytes):,} bytes video returned")
    return Response(content=video_bytes, media_type="video/mp4")
