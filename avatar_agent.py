"""
AI Avatar Agent — Streaming Conversational Pipeline
LiveKit voice + video agent with MuseTalk lip-sync.

Architecture (real-time streaming):
  User audio → STT (AssemblyAI) → LLM (Groq streaming) → TTS (VibeVoice)
                                                              ↓
                                          yield audio frames IMMEDIATELY (user hears in <1s)
                                                              ↓
                                          buffer ALL PCM during TTS stream
                                                              ↓
                                          send all audio to MuseTalk /synthesize_stream
                                                              ↓
                                          server processes with GPU lock + 256x256 fast blend
                                                              ↓
                                          streaming response: frames arrive progressively
                                                              ↓
                                          publisher plays frames one-shot as they arrive
                                                              ↓
                                          all played → return to idle

  VAD / interruptions:
    - LiveKit agents framework handles VAD natively
    - When user speaks mid-response, tts_node is re-entered
    - Running MuseTalk HTTP stream is cancelled
    - Video immediately returns to idle
"""

import asyncio
import io
import json
import logging
import os
import tempfile
import wave
from typing import AsyncGenerator

import cv2
import httpx
import numpy as np

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import assemblyai, openai

import config
from vibevoice_tts import VibeVoiceTTS

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("avatar-agent")

# ── Video constants ───────────────────────────────────────────────────────────
FRAME_W = 256
FRAME_H = 256
FPS = 25
FRAME_INTERVAL = 1.0 / FPS

# TTS audio parameters (VibeVoice outputs 24kHz 16-bit mono)
TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1
TTS_BYTES_PER_SAMPLE = 2


# ── Utilities ─────────────────────────────────────────────────────────────────

def pcm_to_wav(pcm: bytes, rate: int = TTS_SAMPLE_RATE, ch: int = TTS_CHANNELS) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(TTS_BYTES_PER_SAMPLE)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def extract_frames(video_path: str) -> list[np.ndarray]:
    """
    Read every frame from a video, resize to FRAME_W×FRAME_H, return as
    list of RGBA uint8 arrays (shape H×W×4).
    """
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ok, bgr = cap.read()
        if not ok:
            break
        rgba = cv2.cvtColor(cv2.resize(bgr, (FRAME_W, FRAME_H)), cv2.COLOR_BGR2RGBA)
        frames.append(rgba.copy())
    cap.release()
    return frames


def lk_frame(arr: np.ndarray) -> rtc.VideoFrame:
    return rtc.VideoFrame(
        width=FRAME_W,
        height=FRAME_H,
        type=rtc.VideoBufferType.RGBA,
        data=bytearray(arr.tobytes()),
    )


def load_idle_frame(image_path: str) -> rtc.VideoFrame:
    """Load a portrait image as a static LiveKit VideoFrame."""
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Avatar image not found: {image_path} — using grey placeholder")
        img = np.full((FRAME_H, FRAME_W, 3), 100, dtype=np.uint8)
    rgba = cv2.cvtColor(cv2.resize(img, (FRAME_W, FRAME_H)), cv2.COLOR_BGR2RGBA)
    return lk_frame(rgba)


# ── Video publisher (streaming) ────────────────────────────────────────────────

class AvatarVideoPublisher:
    """
    Continuously pushes frames to a LiveKit VideoSource.

    Supports PROGRESSIVE frame delivery:
      - begin_speaking() → enter speaking state (frames may arrive later)
      - append_frames()  → add new lip-sync frames to playback queue
      - mark_done()      → signal no more frames coming
      - stop_speaking()  → immediately return to idle (interruption)

    Frame loop behavior:
      idle     → shows static avatar photo at FPS
      speaking → plays frames once in order as they arrive:
                 - if play_idx < available frames → show next frame
                 - if waiting for more frames → hold last frame (natural pause)
                 - if all done + all played → return to idle
    """

    def __init__(
        self,
        source: rtc.VideoSource,
        avatar_image: str,
        musetalk_url: str,
    ) -> None:
        self._source = source
        self._avatar_image = avatar_image
        self._musetalk_url = musetalk_url.rstrip("/")

        self._idle_frame: rtc.VideoFrame | None = None

        # Progressive playback state
        self._speaking_frames: list[np.ndarray] = []
        self._play_idx = 0
        self._is_speaking = False
        self._all_chunks_done = False
        self._loop_task: asyncio.Task | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self._idle_frame = load_idle_frame(self._avatar_image)
        self._loop_task = asyncio.create_task(self._frame_loop())
        logger.info("AvatarVideoPublisher started")

    def stop(self) -> None:
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()

    # ── speaking state management ─────────────────────────────────────────

    def begin_speaking(self) -> None:
        """Enter speaking state. Frames will be appended progressively."""
        self._speaking_frames = []
        self._play_idx = 0
        self._is_speaking = True
        self._all_chunks_done = False
        logger.debug("Speaking state: started (waiting for frames)")

    def append_frames(self, frames: list[np.ndarray]) -> None:
        """Add new lip-sync frames to the playback queue (thread-safe in asyncio)."""
        self._speaking_frames.extend(frames)
        logger.info(
            f"Appended {len(frames)} frames "
            f"(total: {len(self._speaking_frames)}, played: {self._play_idx})"
        )

    def mark_done(self) -> None:
        """Signal that no more frames will be appended."""
        self._all_chunks_done = True
        logger.debug(
            f"All chunks received ({len(self._speaking_frames)} total frames)"
        )

    def stop_speaking(self) -> None:
        """Immediately return to idle (used for interruptions)."""
        self._is_speaking = False
        self._speaking_frames = []
        self._play_idx = 0
        self._all_chunks_done = False

    # ── MuseTalk communication ────────────────────────────────────────────

    async def send_audio_stream(self, audio_bytes: bytes) -> bool:
        """
        Send ALL PCM audio to MuseTalk /synthesize_stream endpoint.
        Reads progressive RGBA frames from the streaming response and
        appends them to the playback queue as they arrive.
        Returns True on success.
        """
        try:
            wav = pcm_to_wav(audio_bytes)
            logger.info(f"Streaming {len(wav):,} bytes to MuseTalk…")
            t0 = asyncio.get_event_loop().time()

            frame_w = FRAME_W
            frame_h = FRAME_H
            frame_size = frame_w * frame_h * 4  # RGBA
            total_frames = 0
            buffer = b""

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._musetalk_url}/synthesize_stream",
                    files={"audio": ("audio.wav", wav, "audio/wav")},
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.error(
                            f"MuseTalk returned {resp.status_code}: "
                            f"{body[:200]}"
                        )
                        return False

                    # Read 8-byte header
                    header_buf = b""
                    async for raw in resp.aiter_bytes():
                        header_buf += raw
                        if len(header_buf) >= 8:
                            break

                    import struct
                    fw, fh, fps, _ = struct.unpack("<HHHH", header_buf[:8])
                    frame_w = fw or FRAME_W
                    frame_h = fh or FRAME_H
                    frame_size = frame_w * frame_h * 4
                    buffer = header_buf[8:]

                    # Stream frames progressively
                    async for raw_chunk in resp.aiter_bytes():
                        buffer += raw_chunk

                        # Extract complete frames from buffer
                        batch = []
                        while len(buffer) >= frame_size:
                            arr = np.frombuffer(
                                buffer[:frame_size], dtype=np.uint8
                            )
                            arr = arr.reshape(frame_h, frame_w, 4)
                            batch.append(arr.copy())
                            buffer = buffer[frame_size:]

                        if batch:
                            self.append_frames(batch)
                            total_frames += len(batch)

            elapsed = asyncio.get_event_loop().time() - t0
            logger.info(
                f"Stream done: {total_frames} frames in {elapsed:.1f}s "
                f"({total_frames/max(elapsed,0.001):.0f} fps)"
            )

            self.mark_done()
            return total_frames > 0

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"send_audio_stream failed: {exc}")
            return False

    async def send_audio_bulk(self, audio_bytes: bytes) -> bool:
        """
        Fallback: send ALL audio to /synthesize_frames (non-streaming).
        Used if streaming endpoint is unavailable.
        """
        try:
            wav = pcm_to_wav(audio_bytes)
            logger.info(f"Bulk sending {len(wav):,} bytes to MuseTalk…")
            t0 = asyncio.get_event_loop().time()

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._musetalk_url}/synthesize_frames",
                    files={"audio": ("audio.wav", wav, "audio/wav")},
                )

            elapsed = asyncio.get_event_loop().time() - t0

            if resp.status_code == 200 and "X-Frame-Count" in resp.headers:
                frame_count = int(resp.headers["X-Frame-Count"])
                frame_w = int(resp.headers.get("X-Frame-Width", "256"))
                frame_h = int(resp.headers.get("X-Frame-Height", "256"))
                raw_data = resp.content
                frame_size = frame_w * frame_h * 4

                new_frames = []
                for i in range(frame_count):
                    offset = i * frame_size
                    if offset + frame_size > len(raw_data):
                        break
                    arr = np.frombuffer(
                        raw_data[offset : offset + frame_size], dtype=np.uint8
                    )
                    arr = arr.reshape(frame_h, frame_w, 4)
                    new_frames.append(arr.copy())

                self.append_frames(new_frames)
                self.mark_done()
                logger.info(
                    f"Bulk done: {len(new_frames)} frames in {elapsed:.1f}s"
                )
                return True
            else:
                logger.error(
                    f"MuseTalk returned {resp.status_code}: {resp.text[:200]}"
                )
                return False

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"send_audio_bulk failed: {exc}")
            return False

    # ── internal frame loop ────────────────────────────────────────────────

    async def _frame_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            t0 = loop.time()

            if self._is_speaking:
                n = len(self._speaking_frames)

                if self._play_idx < n:
                    # Have a frame to show → play it
                    arr = self._speaking_frames[self._play_idx]
                    self._source.capture_frame(lk_frame(arr))
                    self._play_idx += 1

                elif self._all_chunks_done:
                    # All frames played and no more coming → back to idle
                    self._is_speaking = False
                    self._play_idx = 0
                    logger.info("Lip-sync playback complete → idle")
                    if self._idle_frame is not None:
                        self._source.capture_frame(self._idle_frame)

                elif n > 0:
                    # Waiting for more frames → hold the last frame (natural pause)
                    arr = self._speaking_frames[-1]
                    self._source.capture_frame(lk_frame(arr))

                else:
                    # Speaking but no frames yet → show idle
                    if self._idle_frame is not None:
                        self._source.capture_frame(self._idle_frame)

            elif self._idle_frame is not None:
                self._source.capture_frame(self._idle_frame)

            elapsed = loop.time() - t0
            await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))


# ── Agent ──────────────────────────────────────────────────────────────────────

class AvatarAgent(Agent):
    """
    Voice + video agent with REAL-TIME lip-sync pipeline.

    Architecture:
      1. Yield TTS audio frames IMMEDIATELY → user hears response in <1s
      2. Buffer ALL raw PCM as TTS streams
      3. When TTS is done, send all audio to MuseTalk /synthesize_stream
      4. Server returns frames progressively (streaming response)
      5. Publisher plays frames one-shot as they arrive
      6. On interruption: cancel the streaming task, stop video → idle

    With v3 server (256x256 blending + GPU lock + large batch):
      - Audio latency:      ~1s (TTS first frame)
      - First lip-sync:     ~2-3s after TTS ends
      - Frame rate:         real-time (GPU processes faster than playback)
    """

    def __init__(self, publisher: AvatarVideoPublisher, **kwargs) -> None:
        super().__init__(**kwargs)
        self._publisher = publisher
        self._lipsync_task: asyncio.Task | None = None

    # ── TTS node override ──────────────────────────────────────────────────

    async def tts_node(
        self,
        text,               # AsyncIterable[str] from LLM
        model_settings: ModelSettings,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:

        # Cancel any previous lip-sync
        self._cancel_lipsync()

        # Enter speaking state (publisher will hold idle until frames arrive)
        self._publisher.begin_speaking()

        raw_chunks: list[bytes] = []

        # ── Yield audio IMMEDIATELY as TTS produces it ────────────────────
        async for frame in Agent.default.tts_node(self, text, model_settings):
            raw_chunks.append(bytes(frame.data))
            yield frame   # ← user hears this right away!

        if not raw_chunks:
            self._publisher.stop_speaking()
            return

        # ── Send ALL audio to MuseTalk for streaming lip-sync ─────────────
        audio_bytes = b"".join(raw_chunks)
        audio_duration = len(audio_bytes) / (TTS_SAMPLE_RATE * TTS_CHANNELS * TTS_BYTES_PER_SAMPLE)
        logger.info(
            f"TTS done: {len(audio_bytes):,} bytes, "
            f"{audio_duration:.1f}s — streaming to MuseTalk"
        )

        self._lipsync_task = asyncio.create_task(
            self._stream_lipsync(audio_bytes)
        )

    async def _stream_lipsync(self, audio_bytes: bytes) -> None:
        """Background: stream lip-sync frames from MuseTalk v3 server."""
        try:
            success = await self._publisher.send_audio_stream(audio_bytes)
            if not success:
                # Fallback to bulk endpoint
                logger.warning("Streaming failed, trying bulk endpoint…")
                success = await self._publisher.send_audio_bulk(audio_bytes)

            if not success:
                logger.warning("Lip-sync generation failed")
                self._publisher.stop_speaking()

        except asyncio.CancelledError:
            self._publisher.stop_speaking()
            logger.info("Lip-sync stream cancelled (interruption)")
        except Exception as exc:
            logger.error(f"Lip-sync error: {exc}")
            self._publisher.stop_speaking()

    def _cancel_lipsync(self) -> None:
        """Cancel any running lip-sync task, return to idle."""
        if self._lipsync_task and not self._lipsync_task.done():
            self._lipsync_task.cancel()
            logger.info("Cancelled lip-sync task")
        self._lipsync_task = None
        self._publisher.stop_speaking()


# ── Entry point ────────────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    logger.info(f"Joining room: {ctx.room.name}")

    # Audio-only subscribe (we publish video ourselves)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for user to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # ── Read agent config from participant metadata ────────────────────────
    metadata: dict = {}
    if participant.metadata:
        try:
            metadata = json.loads(participant.metadata)
        except Exception:
            pass

    system_prompt = metadata.get(
        "instructions",
        "You are a helpful conversational assistant. Be concise and friendly.",
    )
    greeting = metadata.get("greeting", "Hello! How can I help you today?")
    tts_voice = metadata.get("ttsVoiceId", config.VIBEVOICE_VOICE)

    # ── Setup LiveKit video track ──────────────────────────────────────────
    video_source = rtc.VideoSource(width=FRAME_W, height=FRAME_H)
    video_track = rtc.LocalVideoTrack.create_video_track("avatar-video", video_source)
    video_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await ctx.room.local_participant.publish_track(video_track, video_opts)
    logger.info("Video track published")

    # ── Start avatar video publisher ───────────────────────────────────────
    publisher = AvatarVideoPublisher(
        source=video_source,
        avatar_image=config.AVATAR_IMAGE,
        musetalk_url=config.MUSETALK_SERVER_URL,
    )
    publisher.start()

    # ── STT ───────────────────────────────────────────────────────────────
    stt_instance = assemblyai.STT(
        api_key=config.ASSEMBLYAI_API_KEY,
        sample_rate=16000,
    )

    # ── LLM ───────────────────────────────────────────────────────────────
    logger.info(f"LLM: {config.SELFHOSTED_LLM_MODEL} @ {config.SELFHOSTED_LLM_URL}")
    llm_instance = openai.LLM(
        model=config.SELFHOSTED_LLM_MODEL,
        base_url=config.SELFHOSTED_LLM_URL,
        api_key=config.SELFHOSTED_LLM_API_KEY,
        temperature=0.1,
        max_completion_tokens=80,
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
    )

    # ── TTS ───────────────────────────────────────────────────────────────
    logger.info(f"TTS: VibeVoice voice={tts_voice} @ {config.VIBEVOICE_TTS_URL}")
    tts_instance = VibeVoiceTTS(
        base_url=config.VIBEVOICE_TTS_URL,
        voice=tts_voice,
        speed=1.3,
    )

    # ── Build agent + session ─────────────────────────────────────────────
    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(role="system", content=system_prompt)

    agent = AvatarAgent(
        publisher=publisher,
        instructions=system_prompt,
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
        chat_ctx=initial_ctx,
    )

    llm_opts = APIConnectOptions(timeout=30.0, max_retry=1, retry_interval=0.5)
    tts_opts = APIConnectOptions(timeout=15.0, max_retry=1, retry_interval=0.5)
    session = AgentSession(
        conn_options=SessionConnectOptions(
            llm_conn_options=llm_opts,
            tts_conn_options=tts_opts,
        ),
        min_endpointing_delay=0.1,
        max_endpointing_delay=0.6,
        preemptive_generation=True,
        min_interruption_duration=0.3,
    )

    await session.start(agent, room=ctx.room)
    await session.say(greeting, allow_interruptions=True)
    logger.info("Avatar agent started and greeting sent")

    # Keep alive until room disconnects
    done = asyncio.Event()

    @session.on("close")
    def _on_close(*_):
        done.set()

    @ctx.room.on("disconnected")
    def _on_disconnected(*_):
        done.set()

    try:
        await done.wait()
    except asyncio.CancelledError:
        pass
    finally:
        publisher.stop()
        logger.info("Avatar agent stopped")


# ── Worker ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not config.validate():
        import sys
        sys.exit(1)

    logger.info(f"LiveKit URL : {config.LIVEKIT_URL}")
    logger.info(f"LLM        : {config.SELFHOSTED_LLM_MODEL} @ {config.SELFHOSTED_LLM_URL}")
    logger.info(f"TTS        : {config.VIBEVOICE_VOICE} @ {config.VIBEVOICE_TTS_URL}")
    logger.info(f"MuseTalk   : {config.MUSETALK_SERVER_URL}")
    logger.info(f"Avatar     : {config.AVATAR_IMAGE}")

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="avatar-agent",
            num_idle_processes=0,
        )
    )


if __name__ == "__main__":
    main()
