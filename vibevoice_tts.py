"""
VibeVoice TTS Plugin for LiveKit Agents

A custom TTS plugin that connects to a self-hosted VibeVoice server
(OpenAI-compatible API) with true real-time streaming.

With stream=true, VibeVoice streams PCM audio chunks as they are generated
on the GPU, giving ~0.6s time-to-first-audio instead of waiting for the
entire utterance to be synthesized.
"""

import logging
import re
import time

import httpx
from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.types import NOT_GIVEN, NotGivenOr


def expressify(text: str) -> str:
    """
    Preprocess text for expressive speech synthesis with Kokoro TTS.

    Kokoro 0.5B renders emotion through punctuation, pacing cues (ellipses,
    dashes, commas) and natural interjections. This function:
      1. Strips artifacts the LLM may produce that should NOT be spoken
         (markdown, asterisk actions, bracket stage directions, emojis).
      2. Converts written cues into Kokoro-friendly punctuation that
         actually changes the prosody of the generated speech.
    """
    if not text:
        return text

    # ── 1. Remove artifacts that would be spoken literally ──────────────
    # Strip markdown bold/italic markers: *word*, **word**, _word_
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', text)

    # Strip bracketed / parenthesised stage directions: [laughs], (sighs)
    text = re.sub(r'[\[\(][^\]\)]{0,30}[\]\)]', '', text)

    # Strip emojis (Unicode emoji ranges)
    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F900-\U0001F9FF\U00002702-\U000027B0\U0001FA00-\U0001FA6F'
        r'\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0000200D]+', '', text
    )

    # Strip hash headers: "## Title" -> "Title"
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Strip bullet points / numbered list markers at line start
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+[.):]\s+', '', text, flags=re.MULTILINE)

    # ── 2. Convert written cues into expressive punctuation ────────────
    # "ha ha" / "haha" -> natural laugh cue
    text = re.sub(r'\b(ha\s*){2,}', 'ha ha, ', text, flags=re.IGNORECASE)

    # Ensure exclamations and questions get proper punctuation (drives prosody)
    # Double-check sentences don't end bare
    text = re.sub(r'(?<=[a-zA-Z])\s*$', '.', text.rstrip())

    # Collapse multiple spaces / newlines into single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Collapse repeated punctuation: "!!!" -> "!", "..." stays (pacing)
    text = re.sub(r'([!?]){2,}', r'\1', text)

    return text

logger = logging.getLogger("vibevoice-tts")

# VibeVoice outputs 24kHz 16-bit mono PCM
SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# Voice mapping: OpenAI name -> VibeVoice speaker
VOICE_MAP = {
    "alloy": "Carter",     # male
    "echo": "Davis",       # male
    "fable": "Emma",       # female
    "onyx": "Frank",       # male
    "nova": "Grace",       # female
    "shimmer": "Mike",     # male
}


class VibeVoiceTTS(tts.TTS):
    """LiveKit TTS plugin for self-hosted VibeVoice with true streaming."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:9000",
        voice: str = "alloy",
        model: str = "tts-1",
        speed: float = 1.5,
        response_format: str = "pcm",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        # Ensure base_url ends with /v1
        self._base_url = base_url.rstrip("/")
        if not self._base_url.endswith("/v1"):
            self._base_url += "/v1"

        self._voice = voice
        self._model = model
        self._speed = speed
        self._response_format = response_format

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        )

        speaker = VOICE_MAP.get(voice, voice)
        logger.info(
            f"🔊 VibeVoice TTS initialized — server: {self._base_url}, "
            f"voice: {voice} ({speaker}), format: {response_format}, "
            f"sample_rate: {SAMPLE_RATE}Hz, streaming: enabled"
        )

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if voice is not NOT_GIVEN:
            self._voice = voice
            logger.info(f"🔊 VibeVoice voice changed to: {voice} ({VOICE_MAP.get(voice, voice)})")
        if speed is not NOT_GIVEN:
            self._speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "VibeVoiceChunkedStream":
        return VibeVoiceChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class VibeVoiceChunkedStream(tts.ChunkedStream):
    """Streams audio from VibeVoice with progressive chunk delivery."""

    def __init__(
        self,
        *,
        tts: VibeVoiceTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._vv_tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        url = f"{self._vv_tts._base_url}/audio/speech"
        processed_text = expressify(self.input_text)
        payload = {
            "input": processed_text,
            "voice": self._vv_tts._voice,
            "model": self._vv_tts._model,
            "response_format": self._vv_tts._response_format,
            "speed": self._vv_tts._speed,
            "stream": True,
        }

        text_preview = self.input_text[:80] + ("..." if len(self.input_text) > 80 else "")
        speaker = VOICE_MAP.get(self._vv_tts._voice, self._vv_tts._voice)
        logger.info(f"🔊 VibeVoice TTS synthesizing [{speaker}]: \"{text_preview}\"")

        t_start = time.monotonic()
        t_first_chunk = None
        total_bytes = 0
        chunk_count = 0

        try:
            async with self._vv_tts._client.stream(
                "POST",
                url,
                json=payload,
                timeout=httpx.Timeout(
                    60.0, connect=self._conn_options.timeout
                ),
            ) as response:
                response.raise_for_status()

                output_emitter.initialize(
                    request_id=f"vibevoice-{id(self)}",
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for chunk in response.aiter_bytes():
                    if t_first_chunk is None:
                        t_first_chunk = time.monotonic() - t_start
                        logger.info(
                            f"🔊 VibeVoice first audio chunk in {t_first_chunk:.2f}s "
                            f"({len(chunk)} bytes) — streaming to speaker"
                        )

                    output_emitter.push(chunk)
                    total_bytes += len(chunk)
                    chunk_count += 1

            output_emitter.flush()

            t_total = time.monotonic() - t_start
            duration_s = total_bytes / (SAMPLE_RATE * 2)  # 16-bit = 2 bytes/sample
            logger.info(
                f"🔊 VibeVoice done: {duration_s:.1f}s audio in {chunk_count} chunks, "
                f"TTFA={t_first_chunk:.2f}s, total={t_total:.2f}s"
            )

        except httpx.TimeoutException:
            logger.error("🔊 VibeVoice TTS timeout")
            raise tts.APITimeoutError() from None
        except httpx.HTTPStatusError as e:
            logger.error(f"🔊 VibeVoice TTS HTTP error: {e.response.status_code}")
            raise tts.APIStatusError(
                str(e),
                status_code=e.response.status_code,
                request_id=None,
                body=e.response.text,
            ) from None
        except Exception as e:
            logger.error(f"🔊 VibeVoice TTS connection error: {e}")
            raise tts.APIConnectionError() from e
