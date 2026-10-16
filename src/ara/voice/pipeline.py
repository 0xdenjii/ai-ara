"""Voice pipeline — main orchestration loop.

Coordinates all voice components into a single pipeline:
    Mic -> Wake Word/Hotkey -> VAD -> STT -> [LLM callback] -> TTS -> Speaker

The pipeline runs as an async loop, transitioning between states:
    IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE

The LLM is not owned by this module. Instead, the caller provides an
async callback (on_transcription) that receives transcribed text and
returns the assistant's response. This keeps voice and LLM decoupled.

All configuration comes from AraSettings.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ara.utils.errors import VoiceError
from ara.utils.logging import get_logger
from ara.utils.types import VoicePipelineState
from ara.voice.audio import AudioRingBuffer, MicrophoneStream, SpeakerOutput
from ara.voice.stt import STTProcessor
from ara.voice.tts import TTSProcessor
from ara.voice.vad import SpeechSegmentTracker, VADProcessor
from ara.voice.wake_word import WakeWordDetector

if TYPE_CHECKING:
    from ara.config.settings import AraSettings

log = get_logger("pipeline")

# Type alias for the LLM callback:
# Takes transcribed user text, returns assistant response text.
# The callback receives an on_sentence callback for streaming TTS.
TranscriptionCallback = Callable[
    [str, Callable[[str], Coroutine[Any, Any, None]]],
    Coroutine[Any, Any, str],
]


class VoicePipeline:
    """Main voice pipeline orchestrating mic -> STT -> LLM -> TTS -> speaker.

    The pipeline is a state machine:
        IDLE: Waiting for wake word or hotkey activation.
        LISTENING: Mic active, feeding audio through VAD + STT.
        PROCESSING: User stopped speaking; final transcription sent to LLM.
        SPEAKING: Playing TTS audio. Monitors for barge-in.

    Usage:
        pipeline = VoicePipeline(settings)
        pipeline.set_transcription_callback(my_llm_handler)
        await pipeline.start()
        # ... runs until stopped
        await pipeline.stop()
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings

        # Components
        self._mic = MicrophoneStream(settings.audio)
        self._speaker = SpeakerOutput(settings.tts)
        self._vad = VADProcessor(settings.vad)
        self._stt = STTProcessor(settings.stt)
        self._tts = TTSProcessor(settings.tts)
        self._wake_word = WakeWordDetector(settings.wake_word)
        self._ring_buffer = AudioRingBuffer()
        self._segment_tracker = SpeechSegmentTracker(settings.vad, settings.audio.sample_rate)

        # State
        self._state = VoicePipelineState.IDLE
        self._running = False
        self._activated = asyncio.Event()

        # Runtime toggles (override settings without persisting)
        self._wake_word_enabled = settings.wake_word.enabled
        self._hotkey_enabled = True

        # Callbacks
        self._on_transcription: TranscriptionCallback | None = None
        self._on_state_change: Callable[[VoicePipelineState], Any] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Pre-speech audio buffer: holds chunks during VAD onset detection
        # so they can be retroactively fed to STT when speech is confirmed.
        # ~20 chunks at 30ms each = ~600ms lookback (covers typical min_speech_ms)
        self._pre_speech_buffer: deque[Any] = deque(maxlen=20)

        # Metrics
        self._utterance_start_time: float = 0.0
        self._total_utterances: int = 0

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def state(self) -> VoicePipelineState:
        """Current pipeline state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Whether the pipeline loop is active."""
        return self._running

    @property
    def total_utterances(self) -> int:
        """Number of utterances processed since start."""
        return self._total_utterances

    @property
    def wake_word_enabled(self) -> bool:
        """Whether wake word detection is enabled at runtime."""
        return self._wake_word_enabled

    @wake_word_enabled.setter
    def wake_word_enabled(self, value: bool) -> None:
        self._wake_word_enabled = value
        log.info("wake_word_toggled", enabled=value)

    @property
    def hotkey_enabled(self) -> bool:
        """Whether hotkey activation is enabled at runtime."""
        return self._hotkey_enabled

    @hotkey_enabled.setter
    def hotkey_enabled(self, value: bool) -> None:
        self._hotkey_enabled = value
        log.info("hotkey_toggled", enabled=value)

    # ── Configuration ────────────────────────────────────────────────────

    def set_transcription_callback(self, callback: TranscriptionCallback) -> None:
        """Set the callback invoked when a complete utterance is transcribed.

        The callback signature is:
            async def handler(text: str, on_sentence: async (str) -> None) -> str

        Args:
            callback: Async function that takes the user's transcribed text
                      and a sentence callback for streaming TTS, then returns
                      the full assistant response.
        """
        self._on_transcription = callback

    def set_state_change_callback(self, callback: Callable[[VoicePipelineState], Any]) -> None:
        """Set a callback invoked whenever the pipeline state changes.

        Useful for UI updates (show listening indicator, speaking animation, etc.).
        """
        self._on_state_change = callback

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Load all models and start the pipeline loop.

        Raises:
            VoiceError: If any component fails to load.
        """
        if self._running:
            log.warning("pipeline_already_running")
            return

        log.info("pipeline_starting")
        load_start = time.monotonic()

        # Load models (all CPU, no VRAM)
        await self._load_models()

        # Set up the ring buffer event loop for cross-thread signaling
        loop = asyncio.get_running_loop()
        self._ring_buffer.set_event_loop(loop)

        # Start mic capture
        self._mic.start(self._ring_buffer)

        load_ms = (time.monotonic() - load_start) * 1000
        log.info("pipeline_started", load_time_ms=round(load_ms, 1))

        self._running = True
        self._set_state(VoicePipelineState.IDLE)

    async def stop(self) -> None:
        """Stop the pipeline and release all resources."""
        if not self._running:
            return

        log.info("pipeline_stopping")
        self._running = False
        self._activated.set()  # unblock any pending wait

        # Stop mic
        self._mic.stop()

        # Stop any active playback
        self._speaker.stop()

        # Unload models
        self._vad.unload()
        self._stt.unload()
        self._tts.unload()
        self._wake_word.unload()

        # Clear buffers
        self._ring_buffer.clear()
        self._segment_tracker.reset()

        self._set_state(VoicePipelineState.IDLE)
        log.info(
            "pipeline_stopped",
            total_utterances=self._total_utterances,
        )

    async def run(self) -> None:
        """Run the pipeline loop until stopped.

        This is the main entry point. Call start() first, then run().
        The loop processes audio continuously until stop() is called.

        Usage:
            await pipeline.start()
            await pipeline.run()  # blocks until stopped
        """
        if not self._running:
            raise VoiceError(
                "Pipeline not started — call start() first",
                operation="run",
            )

        log.info("pipeline_loop_started")

        try:
            while self._running:
                if self._state == VoicePipelineState.IDLE:
                    await self._handle_idle()
                elif self._state == VoicePipelineState.LISTENING:
                    await self._handle_listening()
                elif self._state == VoicePipelineState.PROCESSING:
                    await self._handle_processing()
                elif self._state == VoicePipelineState.SPEAKING:
                    await self._handle_speaking()
        except asyncio.CancelledError:
            log.info("pipeline_loop_cancelled")
        except Exception as exc:
            log.error("pipeline_loop_error", error=str(exc), exc_info=True)
            raise VoiceError(
                f"Pipeline loop crashed: {exc}",
                operation="run",
                details={"state": self._state.value},
            ) from exc
        finally:
            log.info("pipeline_loop_exited")

    def activate(self) -> None:
        """Manually activate the pipeline (hotkey trigger).

        Skips wake word detection and immediately transitions to LISTENING.
        Safe to call from any thread.
        """
        if self._state == VoicePipelineState.IDLE:
            log.info("pipeline_manual_activation")
            self._activated.set()

    # ── State Handlers ───────────────────────────────────────────────────

    async def _handle_idle(self) -> None:
        """IDLE state: wait for wake word detection or manual activation."""
        # Process audio for wake word detection
        has_data = await self._ring_buffer.wait_for_data(timeout=0.1)
        if not has_data or not self._running:
            return

        # Check for manual activation first
        if self._activated.is_set():
            self._activated.clear()
            self._transition_to_listening()
            return

        # Process buffered audio for wake word
        chunk = self._ring_buffer.get()
        if chunk is None:
            return

        if not self._wake_word_enabled or not self._wake_word.is_enabled:
            # Wake word disabled (runtime toggle or settings) — stay idle
            # until manual activation. Drain buffer to prevent buildup
            self._ring_buffer.clear()
            await asyncio.sleep(0.05)
            return

        if not self._wake_word.is_loaded:
            return

        result = self._wake_word.process(chunk)
        if result.detected:
            log.info(
                "wake_word_activated",
                model=result.model_name,
                confidence=result.confidence,
            )
            self._transition_to_listening()

    async def _handle_listening(self) -> None:
        """LISTENING state: feed audio through VAD + STT.

        Continuously processes mic audio. When silence is detected after
        speech, transitions to PROCESSING with the final transcription.
        """
        has_data = await self._ring_buffer.wait_for_data(timeout=0.1)
        if not has_data or not self._running:
            return

        chunk = self._ring_buffer.get()
        if chunk is None:
            return

        # Run VAD on the chunk
        vad_result = self._vad.process(chunk, sample_rate=self._settings.audio.sample_rate)
        state_changed = self._segment_tracker.update(
            is_speech=vad_result.is_speech,
            chunk_samples=len(chunk),
        )

        if self._segment_tracker.is_speaking:
            # On speech onset, flush pre-speech buffer to STT first
            if state_changed:
                for buffered_chunk in self._pre_speech_buffer:
                    self._stt.feed_audio(
                        buffered_chunk.tolist(), sample_rate=self._settings.audio.sample_rate
                    )
                self._pre_speech_buffer.clear()

            # Feed current audio to STT as float list (moonshine-voice native format)
            self._stt.feed_audio(chunk.tolist(), sample_rate=self._settings.audio.sample_rate)

        else:
            # Not yet confirmed as speech — buffer for retroactive STT feed
            self._pre_speech_buffer.append(chunk)

        if state_changed and not self._segment_tracker.is_speaking:
            # Speech just ended — get final transcription
            self._pre_speech_buffer.clear()
            event = self._stt.end_utterance()
            if event is not None and event.text.strip():
                elapsed_ms = (time.monotonic() - self._utterance_start_time) * 1000
                log.info(
                    "utterance_captured",
                    text_length=len(event.text),
                    listen_time_ms=round(elapsed_ms, 1),
                )
                self._set_state(VoicePipelineState.PROCESSING)
                await self._process_utterance(event.text)
            else:
                log.debug("empty_utterance_discarded")
                # Stay in listening for another attempt
                self._stt.reset()
                self._stt.start_utterance()

    async def _handle_processing(self) -> None:
        """PROCESSING state: waiting for LLM response.

        This state is transient — _process_utterance handles the transition
        to SPEAKING or back to IDLE. If we end up here in the loop,
        just yield control.
        """
        await asyncio.sleep(0.01)

    async def _handle_speaking(self) -> None:
        """SPEAKING state: playing TTS audio, monitoring for barge-in.

        Checks if the user starts speaking (VAD detects speech) during
        playback. If so, interrupts TTS and transitions to LISTENING.
        """
        if not self._speaker.is_playing:
            # Playback finished naturally
            self._set_state(VoicePipelineState.IDLE)
            return

        # Check for barge-in: process mic audio through VAD during playback
        has_data = await self._ring_buffer.wait_for_data(timeout=0.05)
        if has_data:
            chunk = self._ring_buffer.get()
            if chunk is not None:
                vad_result = self._vad.process(chunk, sample_rate=self._settings.audio.sample_rate)
                if vad_result.is_speech and vad_result.confidence >= self._settings.vad.threshold:
                    log.info(
                        "barge_in_detected",
                        confidence=vad_result.confidence,
                    )
                    self._speaker.stop()
                    self._transition_to_listening()
                    return

        await asyncio.sleep(0.01)

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _transition_to_listening(self) -> None:
        """Set up for listening: reset trackers, start new utterance."""
        self._ring_buffer.clear()
        self._segment_tracker.reset()
        self._vad.reset()
        self._stt.start_utterance()
        self._utterance_start_time = time.monotonic()
        self._set_state(VoicePipelineState.LISTENING)

    async def _process_utterance(self, text: str) -> None:
        """Send transcribed text to LLM and stream TTS response.

        Args:
            text: The user's transcribed utterance.
        """
        self._total_utterances += 1
        process_start = time.monotonic()

        if self._on_transcription is None:
            log.warning("no_transcription_callback", text=text[:50])
            self._set_state(VoicePipelineState.IDLE)
            return

        try:
            # Set up sentence-streaming TTS

            async def on_sentence(sentence: str) -> None:
                """Synthesize and play each sentence as it arrives."""
                if not self._running:
                    return
                try:
                    loop = asyncio.get_running_loop()
                    tts_result = await loop.run_in_executor(None, self._tts.synthesize, sentence)
                    if len(tts_result.samples) > 0:
                        self._set_state(VoicePipelineState.SPEAKING)
                        await self._speaker.play(
                            tts_result.samples,
                            tts_result.sample_rate,
                        )
                except VoiceError as exc:
                    log.warning("tts_sentence_error", error=str(exc))

            # Call the LLM handler with the sentence callback
            response = await self._on_transcription(text, on_sentence)

            process_ms = (time.monotonic() - process_start) * 1000
            log.info(
                "utterance_processed",
                text_length=len(text),
                response_length=len(response),
                total_ms=round(process_ms, 1),
                utterance_number=self._total_utterances,
            )

        except asyncio.CancelledError:
            log.info("utterance_processing_cancelled")
            raise
        except Exception as exc:
            log.error(
                "utterance_processing_error",
                error=str(exc),
                text_preview=text[:50],
            )

        # Return to idle after speaking
        if self._state != VoicePipelineState.LISTENING:
            # Don't override if barge-in already moved us to LISTENING
            self._set_state(VoicePipelineState.IDLE)

    def _set_state(self, new_state: VoicePipelineState) -> None:
        """Update pipeline state and notify listeners."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        log.debug("state_transition", from_state=old_state.value, to_state=new_state.value)

        if self._on_state_change is not None:
            try:
                result = self._on_state_change(new_state)
                # Support async state change callbacks
                if hasattr(result, "__await__"):
                    # Store reference to prevent GC + allow cleanup
                    task = asyncio.ensure_future(result)
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            except Exception as exc:
                log.warning("state_change_callback_error", error=str(exc))

    async def _load_models(self) -> None:
        """Load all voice models (CPU-only, no VRAM).

        Raises:
            VoiceError: If any model fails to load.
        """
        load_times: dict[str, float] = {}

        # Load VAD
        start = time.monotonic()
        self._vad.load()
        load_times["vad"] = (time.monotonic() - start) * 1000

        # Load STT
        start = time.monotonic()
        self._stt.load()
        load_times["stt"] = (time.monotonic() - start) * 1000

        # Load TTS
        start = time.monotonic()
        self._tts.load()
        load_times["tts"] = (time.monotonic() - start) * 1000

        # Load Wake Word (may be disabled)
        start = time.monotonic()
        self._wake_word.load()
        load_times["wake_word"] = (time.monotonic() - start) * 1000

        log.info(
            "models_loaded",
            load_times_ms={k: round(v, 1) for k, v in load_times.items()},
            total_ms=round(sum(load_times.values()), 1),
        )

    # ── Getters for testing / debugging ──────────────────────────────────

    @property
    def vad(self) -> VADProcessor:
        """Access to VAD processor (for testing/debugging)."""
        return self._vad

    @property
    def stt(self) -> STTProcessor:
        """Access to STT processor (for testing/debugging)."""
        return self._stt

    @property
    def tts(self) -> TTSProcessor:
        """Access to TTS processor (for testing/debugging)."""
        return self._tts

    @property
    def wake_word(self) -> WakeWordDetector:
        """Access to wake word detector (for testing/debugging)."""
        return self._wake_word

    @property
    def mic(self) -> MicrophoneStream:
        """Access to microphone stream (for testing/debugging)."""
        return self._mic

    @property
    def speaker(self) -> SpeakerOutput:
        """Access to speaker output (for testing/debugging)."""
        return self._speaker

    @property
    def ring_buffer(self) -> AudioRingBuffer:
        """Access to the audio ring buffer (for testing/debugging)."""
        return self._ring_buffer
