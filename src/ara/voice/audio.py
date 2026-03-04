"""Audio I/O — microphone capture and speaker playback via sounddevice.

Provides a thread-safe ring buffer for continuous audio capture and
async-friendly interfaces for the voice pipeline. All configuration
comes from AraSettings.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from ara.utils.errors import VoiceError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ara.config.settings import AudioSettings, TTSSettings

log = get_logger("audio")


class AudioRingBuffer:
    """Thread-safe ring buffer for audio samples.

    Stores audio in fixed-size chunks. Producers (mic callback) append chunks;
    consumers (VAD/STT) read them. Uses a deque for O(1) append/popleft.
    """

    def __init__(self, max_chunks: int = 300) -> None:
        self._buffer: deque[NDArray[np.float32]] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
        self._event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to an asyncio event loop for cross-thread signaling."""
        self._loop = loop

    def put(self, chunk: NDArray[np.float32]) -> None:
        """Append an audio chunk (called from sounddevice callback thread)."""
        with self._lock:
            self._buffer.append(chunk.copy())

        # Signal async consumers from the callback thread
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._event.set)

    def get(self) -> NDArray[np.float32] | None:
        """Pop the oldest chunk, or None if empty."""
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    def get_all(self) -> list[NDArray[np.float32]]:
        """Drain all chunks from the buffer."""
        with self._lock:
            chunks = list(self._buffer)
            self._buffer.clear()
        return chunks

    def peek(self, n_chunks: int = 1) -> list[NDArray[np.float32]]:
        """Read the last N chunks without removing them."""
        with self._lock:
            return list(self._buffer)[-n_chunks:]

    async def wait_for_data(self, timeout: float = 1.0) -> bool:
        """Async wait until data is available or timeout expires.

        Returns True if data arrived, False on timeout.
        """
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Discard all buffered audio."""
        with self._lock:
            self._buffer.clear()


class MicrophoneStream:
    """Continuous microphone capture using sounddevice.

    Opens an input stream and routes audio chunks into an AudioRingBuffer.
    The stream runs in a separate thread managed by sounddevice.

    Usage:
        mic = MicrophoneStream(audio_settings)
        mic.start(ring_buffer)
        # ... ring_buffer now receives audio chunks
        mic.stop()
    """

    def __init__(self, settings: AudioSettings) -> None:
        self._settings = settings
        self._stream: sd.InputStream | None = None
        self._buffer: AudioRingBuffer | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the microphone is currently capturing."""
        return self._running

    def start(self, buffer: AudioRingBuffer) -> None:
        """Begin capturing audio into the provided ring buffer.

        Raises:
            VoiceError: If the microphone cannot be opened.
        """
        if self._running:
            return

        self._buffer = buffer
        block_size = int(self._settings.sample_rate * self._settings.block_duration_ms / 1000)

        try:
            self._stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="float32",
                blocksize=block_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True

            log.info(
                "microphone_started",
                sample_rate=self._settings.sample_rate,
                channels=self._settings.channels,
                block_size=block_size,
                block_duration_ms=self._settings.block_duration_ms,
            )
        except Exception as exc:
            self._running = False
            raise VoiceError(
                f"Failed to open microphone: {exc}",
                operation="microphone_start",
                details={
                    "sample_rate": self._settings.sample_rate,
                    "channels": self._settings.channels,
                },
            ) from exc

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self._running:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._running = False
        log.info("microphone_stopped")

    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — runs in a separate thread."""
        if status:
            log.warning("audio_callback_status", flags=str(status))

        if self._buffer is not None:
            # indata shape is (frames, channels); squeeze to 1D for mono
            audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            self._buffer.put(audio)


class SpeakerOutput:
    """Audio playback via sounddevice.

    Plays synthesized speech through the default audio output device.
    Supports interruption (barge-in) by stopping playback mid-stream.

    Usage:
        speaker = SpeakerOutput(tts_settings)
        await speaker.play(samples, sample_rate)
        speaker.stop()  # interrupt if needed
    """

    def __init__(self, settings: TTSSettings) -> None:
        self._settings = settings
        self._stream: sd.OutputStream | None = None
        self._playing = False
        self._interrupted = False
        self._play_complete = asyncio.Event()

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently being played."""
        return self._playing

    async def play(self, samples: NDArray[np.float32], sample_rate: int) -> None:
        """Play audio samples through the speaker.

        Args:
            samples: Audio samples as float32 array.
            sample_rate: Sample rate of the audio.

        Raises:
            VoiceError: If playback fails.
        """
        if self._playing:
            self.stop()

        self._playing = True
        self._interrupted = False
        self._play_complete.clear()

        try:
            loop = asyncio.get_running_loop()

            # Run blocking sd.play in executor to avoid blocking the event loop
            await loop.run_in_executor(
                None,
                self._play_blocking,
                samples,
                sample_rate,
            )

        except Exception as exc:
            if not self._interrupted:
                raise VoiceError(
                    f"Speaker playback failed: {exc}",
                    operation="speaker_play",
                    details={"sample_rate": sample_rate, "samples_len": len(samples)},
                ) from exc
        finally:
            self._playing = False
            self._play_complete.set()

    def _play_blocking(self, samples: NDArray[np.float32], sample_rate: int) -> None:
        """Blocking playback — runs in executor thread."""
        sd.play(samples, samplerate=sample_rate)
        sd.wait()

    def stop(self) -> None:
        """Interrupt current playback (for barge-in support)."""
        self._interrupted = True
        sd.stop()
        self._playing = False
        self._play_complete.set()
        log.info("speaker_interrupted")

    async def wait_until_done(self) -> None:
        """Wait until current playback finishes or is interrupted."""
        await self._play_complete.wait()
