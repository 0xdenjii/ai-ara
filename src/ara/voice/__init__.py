"""Voice pipeline — STT, TTS, VAD, wake word, audio I/O, orchestration.

All voice components run on CPU with zero VRAM consumption.
"""

from ara.voice.audio import AudioRingBuffer, MicrophoneStream, SpeakerOutput
from ara.voice.hotkey import HotkeyListener
from ara.voice.pipeline import VoicePipeline
from ara.voice.stt import STTProcessor, TranscriptionEvent
from ara.voice.tts import TTSProcessor, TTSResult
from ara.voice.vad import SpeechSegmentTracker, VADProcessor, VADResult
from ara.voice.wake_word import WakeWordDetector, WakeWordResult

__all__ = [
    "AudioRingBuffer",
    "HotkeyListener",
    "MicrophoneStream",
    "STTProcessor",
    "SpeakerOutput",
    "SpeechSegmentTracker",
    "TTSProcessor",
    "TTSResult",
    "TranscriptionEvent",
    "VADProcessor",
    "VADResult",
    "VoicePipeline",
    "WakeWordDetector",
    "WakeWordResult",
]
