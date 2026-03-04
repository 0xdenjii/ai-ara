"""End-to-end voice pipeline test with latency benchmarking.

Loads all voice components + LLM, then runs an interactive loop:
    Press Ctrl+Space -> speak -> ARA responds via TTS

Measures latency at each stage:
    - Model load times
    - STT latency (speech -> text)
    - LLM TTFT and generation speed
    - TTS synthesis latency (text -> audio)
    - Total end-to-end latency (end of speech -> first audio)

Usage:
    uv run python scripts/test_voice_e2e.py
    uv run python scripts/test_voice_e2e.py --no-llm          # Test voice only (echo mode)
    uv run python scripts/test_voice_e2e.py --hotkey ctrl+alt  # Custom hotkey
    uv run python scripts/test_voice_e2e.py --voice af_sarah   # Different TTS voice
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import time

# Windows-specific event loop policy
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def print_banner() -> None:
    """Print the test banner."""
    print("=" * 60)
    print("  ARA Voice Pipeline — End-to-End Test")
    print("=" * 60)
    print()


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n--- {title} ---\n")


class LatencyTracker:
    """Tracks latency measurements across the pipeline."""

    def __init__(self) -> None:
        self.load_times: dict[str, float] = {}
        self.stt_latencies: list[float] = []
        self.llm_ttft_latencies: list[float] = []
        self.llm_gen_speeds: list[float] = []
        self.tts_latencies: list[float] = []
        self.e2e_latencies: list[float] = []
        self.utterance_count: int = 0

    def print_summary(self) -> None:
        """Print a summary of all latency measurements."""
        print_section("LATENCY SUMMARY")

        print("Model Load Times:")
        for name, ms in self.load_times.items():
            print(f"  {name:>15s}: {ms:>8.1f} ms")
        total_load = sum(self.load_times.values())
        print(f"  {'TOTAL':>15s}: {total_load:>8.1f} ms")

        if self.utterance_count == 0:
            print("\nNo utterances processed yet.")
            return

        print(f"\nUtterances processed: {self.utterance_count}")

        if self.stt_latencies:
            avg_stt = sum(self.stt_latencies) / len(self.stt_latencies)
            print("\nSTT Latency (end_utterance):")
            print(f"  avg: {avg_stt:.1f} ms")

        if self.llm_ttft_latencies:
            avg_ttft = sum(self.llm_ttft_latencies) / len(self.llm_ttft_latencies)
            print("\nLLM TTFT:")
            print(f"  avg: {avg_ttft:.1f} ms")

        if self.llm_gen_speeds:
            avg_speed = sum(self.llm_gen_speeds) / len(self.llm_gen_speeds)
            print("\nLLM Generation Speed:")
            print(f"  avg: {avg_speed:.1f} t/s")

        if self.tts_latencies:
            avg_tts = sum(self.tts_latencies) / len(self.tts_latencies)
            print("\nTTS First-Sentence Latency:")
            print(f"  avg: {avg_tts:.1f} ms")

        if self.e2e_latencies:
            avg_e2e = sum(self.e2e_latencies) / len(self.e2e_latencies)
            print("\nEnd-to-End (speech end -> first audio):")
            print(f"  avg: {avg_e2e:.1f} ms")


async def test_component_loading(args: argparse.Namespace) -> dict[str, object]:
    """Load each voice component individually and report timing."""
    print_section("LOADING COMPONENTS")
    tracker = LatencyTracker()
    components: dict[str, object] = {"tracker": tracker}

    # 1. VAD
    print("Loading Silero VAD...", end=" ", flush=True)
    from ara.voice.vad import VADProcessor
    from ara.config.settings import VADSettings

    vad_settings = VADSettings()
    vad = VADProcessor(vad_settings)
    start = time.monotonic()
    vad.load()
    elapsed = (time.monotonic() - start) * 1000
    tracker.load_times["vad"] = elapsed
    print(f"OK ({elapsed:.0f} ms)")
    components["vad"] = vad

    # 2. STT
    print("Loading Moonshine v2 STT...", end=" ", flush=True)
    from ara.voice.stt import STTProcessor
    from ara.config.settings import STTSettings

    stt_settings = STTSettings()
    stt = STTProcessor(stt_settings)
    start = time.monotonic()
    stt.load()
    elapsed = (time.monotonic() - start) * 1000
    tracker.load_times["stt"] = elapsed
    print(f"OK ({elapsed:.0f} ms)")
    components["stt"] = stt

    # 3. TTS
    voice = args.voice
    print(f"Loading Kokoro TTS (voice: {voice})...", end=" ", flush=True)
    from ara.voice.tts import TTSProcessor
    from ara.config.settings import TTSSettings

    tts_settings = TTSSettings(voice=voice)
    tts = TTSProcessor(tts_settings)
    start = time.monotonic()
    tts.load()
    elapsed = (time.monotonic() - start) * 1000
    tracker.load_times["tts"] = elapsed
    print(f"OK ({elapsed:.0f} ms)")
    components["tts"] = tts

    # 4. Wake Word (optional, skip for hotkey-only test)
    print("Wake word: skipped (using hotkey mode)")

    # 5. LLM (optional)
    if not args.no_llm:
        print("Connecting to Ollama LLM...", end=" ", flush=True)
        from ara.core.llm import LLMClient
        from ara.config.settings import AraSettings

        settings = AraSettings()
        llm = LLMClient(settings)
        start = time.monotonic()
        await llm.connect()
        elapsed = (time.monotonic() - start) * 1000
        tracker.load_times["llm_connect"] = elapsed
        print(f"OK ({elapsed:.0f} ms, model: {llm.active_model})")
        components["llm"] = llm
        components["settings"] = settings
    else:
        print("LLM: skipped (echo mode)")

    # 6. Audio devices
    print("Checking audio devices...", end=" ", flush=True)
    import sounddevice as sd

    default_input = sd.query_devices(kind="input")
    default_output = sd.query_devices(kind="output")
    print("OK")
    print(f"  Input:  {default_input['name']}")  # type: ignore[index]
    print(f"  Output: {default_output['name']}")  # type: ignore[index]

    return components


async def test_tts_synthesis(components: dict[str, object]) -> None:
    """Quick TTS synthesis test — verify audio output works."""
    print_section("TTS SYNTHESIS TEST")

    from ara.voice.tts import TTSProcessor

    tts: TTSProcessor = components["tts"]  # type: ignore[assignment]

    test_text = "Hello! I'm ARA, your personal assistant. Nice to meet you."
    print(f'Synthesizing: "{test_text}"')

    start = time.monotonic()
    result = tts.synthesize(test_text)
    elapsed = (time.monotonic() - start) * 1000
    print(
        f"  Synthesis: {elapsed:.0f} ms, duration: {result.duration_s:.2f}s, "
        f"samples: {len(result.samples)}, rate: {result.sample_rate} Hz"
    )

    print("Playing audio...", end=" ", flush=True)
    import sounddevice as sd

    sd.play(result.samples, samplerate=result.sample_rate)
    sd.wait()
    print("Done")


async def test_vad_quick(components: dict[str, object]) -> None:
    """Quick VAD test — record 2 seconds and check for speech detection."""
    print_section("VAD QUICK TEST")

    from ara.voice.vad import VADProcessor

    vad: VADProcessor = components["vad"]  # type: ignore[assignment]

    import sounddevice as sd

    print("Recording 2 seconds of audio... (speak or stay silent)")
    sample_rate = 16000
    duration = 2.0
    audio = sd.rec(
        int(sample_rate * duration),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio_flat = audio.flatten()

    # Process in chunks of 512 samples (32ms at 16kHz)
    chunk_size = 512
    speech_chunks = 0
    total_chunks = 0

    for i in range(0, len(audio_flat) - chunk_size, chunk_size):
        chunk = audio_flat[i : i + chunk_size]
        result = vad.process(chunk, sample_rate=sample_rate)
        total_chunks += 1
        if result.is_speech:
            speech_chunks += 1

    speech_pct = (speech_chunks / total_chunks * 100) if total_chunks > 0 else 0
    print(f"  Chunks: {total_chunks}, speech: {speech_chunks} ({speech_pct:.0f}%)")
    vad.reset()


async def run_interactive_loop(
    components: dict[str, object],
    hotkey: str,
) -> None:
    """Run the interactive voice loop with Ctrl+Space activation."""
    print_section("INTERACTIVE VOICE LOOP")
    print(f"Press [{hotkey}] to speak. Press Ctrl+C to quit.")
    print("After pressing the hotkey, speak clearly, then pause.")
    print()

    from ara.voice.vad import VADProcessor
    from ara.voice.stt import STTProcessor
    from ara.voice.tts import TTSProcessor
    from ara.voice.audio import AudioRingBuffer, MicrophoneStream, SpeakerOutput
    from ara.voice.vad import SpeechSegmentTracker
    from ara.config.settings import AudioSettings, VADSettings, TTSSettings

    vad: VADProcessor = components["vad"]  # type: ignore[assignment]
    stt: STTProcessor = components["stt"]  # type: ignore[assignment]
    tts: TTSProcessor = components["tts"]  # type: ignore[assignment]
    tracker: LatencyTracker = components["tracker"]  # type: ignore[assignment]

    llm = components.get("llm")
    settings = components.get("settings")

    audio_settings = AudioSettings()
    vad_settings = VADSettings()
    tts_settings = TTSSettings()

    # Set up audio I/O
    ring_buffer = AudioRingBuffer()
    loop = asyncio.get_running_loop()
    ring_buffer.set_event_loop(loop)

    mic = MicrophoneStream(audio_settings)
    speaker = SpeakerOutput(tts_settings)
    segment_tracker = SpeechSegmentTracker(vad_settings, audio_settings.sample_rate)

    # Start mic
    mic.start(ring_buffer)
    print("[MIC] Microphone started")

    # Set up hotkey
    activated = asyncio.Event()

    def on_hotkey() -> None:
        loop.call_soon_threadsafe(activated.set)

    import keyboard

    keyboard.add_hotkey(hotkey, on_hotkey, suppress=False, trigger_on_release=False)
    print(f"[HOTKEY] Registered: {hotkey}")
    print()

    utterance_num = 0

    try:
        while True:
            # Wait for hotkey
            print(f"[IDLE] Waiting for [{hotkey}]...", flush=True)
            activated.clear()
            await activated.wait()
            print("[ACTIVATED] Listening... speak now!")

            # Clear buffers
            ring_buffer.clear()
            segment_tracker.reset()
            vad.reset()
            stt.start_utterance()

            utterance_start = time.monotonic()
            speech_detected = False
            speech_ended = False

            # Listen loop
            while not speech_ended:
                has_data = await ring_buffer.wait_for_data(timeout=0.1)
                if not has_data:
                    # Timeout — if we had speech and now silence, we're done
                    if speech_detected and not segment_tracker.is_speaking:
                        # Extended silence after speech — treat as end
                        pass
                    continue

                chunk = ring_buffer.get()
                if chunk is None:
                    continue

                # VAD
                vad_result = vad.process(chunk, sample_rate=audio_settings.sample_rate)
                state_changed = segment_tracker.update(
                    is_speech=vad_result.is_speech,
                    chunk_samples=len(chunk),
                )

                if segment_tracker.is_speaking:
                    if not speech_detected:
                        speech_detected = True
                        print("  [VAD] Speech detected!", flush=True)

                    # Feed to STT as float list (moonshine-voice native format)
                    event = stt.feed_audio(chunk.tolist(), sample_rate=audio_settings.sample_rate)
                    if event is not None:
                        print(f'  [STT] Partial: "{event.text}"', flush=True)

                elif state_changed and not segment_tracker.is_speaking and speech_detected:
                    speech_ended = True

            # Get final transcription
            speech_end_time = time.monotonic()
            stt_start = time.monotonic()
            final_event = stt.end_utterance()
            stt_elapsed = (time.monotonic() - stt_start) * 1000

            listen_ms = (speech_end_time - utterance_start) * 1000

            if final_event is None or not final_event.text.strip():
                print("  [STT] No speech detected, try again.\n")
                continue

            utterance_num += 1
            tracker.utterance_count = utterance_num
            tracker.stt_latencies.append(stt_elapsed)

            user_text = final_event.text.strip()
            print(f'\n  [STT] Final ({stt_elapsed:.0f} ms): "{user_text}"')
            print(f"  [TIMING] Listen duration: {listen_ms:.0f} ms")

            # Generate response
            response_start = time.monotonic()

            if llm is not None and settings is not None:
                # Real LLM response
                from ara.core.prompts import build_system_prompt
                from ara.core.streaming import StreamBuffer
                from ara.utils.types import ChatMessage, MessageRole

                prompt = build_system_prompt(settings.personality)  # type: ignore[union-attr]
                messages = [
                    ChatMessage(role=MessageRole.SYSTEM, content=prompt),
                    ChatMessage(role=MessageRole.USER, content=user_text + " /no_think"),
                ]

                print("  [LLM] Generating...", flush=True)
                ttft_recorded = False
                token_count = 0
                sentence_buffer = StreamBuffer()
                first_sentence_spoken = False
                response_parts: list[str] = []

                stream = llm.generate_stream(messages)  # type: ignore[union-attr]
                async for token in stream:
                    if not ttft_recorded:
                        ttft_ms = (time.monotonic() - response_start) * 1000
                        tracker.llm_ttft_latencies.append(ttft_ms)
                        ttft_recorded = True
                        print(f"  [LLM] TTFT: {ttft_ms:.0f} ms", flush=True)

                    token_count += 1
                    response_parts.append(token)
                    print(token, end="", flush=True)

                    # Sentence detection for streaming TTS
                    sentences = sentence_buffer.add(token)
                    for sentence in sentences:
                        if not first_sentence_spoken:
                            first_sentence_spoken = True
                            tts_start = time.monotonic()
                            tts_result = tts.synthesize(sentence)
                            tts_ms = (time.monotonic() - tts_start) * 1000
                            tracker.tts_latencies.append(tts_ms)

                            e2e_ms = (time.monotonic() - speech_end_time) * 1000
                            tracker.e2e_latencies.append(e2e_ms)

                            print(
                                f'\n  [TTS] First sentence ({tts_ms:.0f} ms): "{sentence[:50]}..."'
                                if len(sentence) > 50
                                else f'\n  [TTS] First sentence ({tts_ms:.0f} ms): "{sentence}"',
                                flush=True,
                            )
                            print(f"  [E2E] Speech-end -> first audio: {e2e_ms:.0f} ms", flush=True)

                            # Play first sentence
                            await speaker.play(tts_result.samples, tts_result.sample_rate)
                        else:
                            # Synthesize and play remaining sentences
                            tts_result = tts.synthesize(sentence)
                            if len(tts_result.samples) > 0:
                                await speaker.play(tts_result.samples, tts_result.sample_rate)

                # Flush remaining text
                remaining = sentence_buffer.flush()
                if remaining:
                    tts_result = tts.synthesize(remaining)
                    if len(tts_result.samples) > 0:
                        await speaker.play(tts_result.samples, tts_result.sample_rate)

                gen_elapsed = time.monotonic() - response_start
                if gen_elapsed > 0:
                    speed = token_count / gen_elapsed
                    tracker.llm_gen_speeds.append(speed)

                print(f"\n  [LLM] {token_count} tokens in {gen_elapsed:.1f}s ({speed:.1f} t/s)")

            else:
                # Echo mode — no LLM, just TTS the transcription back
                echo_text = f"You said: {user_text}"
                print(f"  [ECHO] {echo_text}")

                tts_start = time.monotonic()
                tts_result = tts.synthesize(echo_text)
                tts_ms = (time.monotonic() - tts_start) * 1000
                tracker.tts_latencies.append(tts_ms)

                e2e_ms = (time.monotonic() - speech_end_time) * 1000
                tracker.e2e_latencies.append(e2e_ms)

                print(f"  [TTS] Synthesis: {tts_ms:.0f} ms, duration: {tts_result.duration_s:.2f}s")
                print(f"  [E2E] Speech-end -> audio: {e2e_ms:.0f} ms")

                await speaker.play(tts_result.samples, tts_result.sample_rate)

            print()

    except KeyboardInterrupt:
        print("\n\n[STOPPING] Ctrl+C received")
    finally:
        mic.stop()
        import contextlib

        with contextlib.suppress(Exception):
            keyboard.remove_hotkey(hotkey)
        print("[MIC] Microphone stopped")

    tracker.print_summary()


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ARA Voice Pipeline E2E Test")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run in echo mode without LLM (just STT -> TTS)",
    )
    parser.add_argument(
        "--hotkey",
        default="ctrl+space",
        help="Hotkey combination to activate listening (default: ctrl+space)",
    )
    parser.add_argument(
        "--voice",
        default="af_bella",
        help="Kokoro TTS voice (default: af_bella)",
    )
    parser.add_argument(
        "--skip-tts-test",
        action="store_true",
        help="Skip the initial TTS playback test",
    )
    parser.add_argument(
        "--skip-vad-test",
        action="store_true",
        help="Skip the initial VAD recording test",
    )
    args = parser.parse_args()

    print_banner()

    # Load components
    components = await test_component_loading(args)

    # Quick TTS test
    if not args.skip_tts_test:
        await test_tts_synthesis(components)

    # Quick VAD test
    if not args.skip_vad_test:
        await test_vad_quick(components)

    # Interactive loop
    await run_interactive_loop(components, args.hotkey)


if __name__ == "__main__":
    asyncio.run(main())
