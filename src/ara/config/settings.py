"""ARA application settings loaded from environment variables and .env files.

All configuration is centralized here. No hardcoded values anywhere in the codebase.
Uses pydantic-settings to validate and type-check all config at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM inference configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_")

    ollama_base_url: str = "http://localhost:11434"
    primary_model: str = "qwen3:30b-a3b"
    long_context_model: str = "qwen3:14b"
    draft_model: str = "qwen3:1b"
    primary_model_context_length: int = 32768
    long_context_length: int = 131072
    temperature: float = 0.8
    max_response_tokens: int = 2048


class AudioSettings(BaseSettings):
    """Audio capture and playback configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_AUDIO_")

    sample_rate: int = 16000
    channels: int = 1
    block_duration_ms: int = 32  # 32ms = 512 samples at 16kHz (matches Silero VAD window)


class STTSettings(BaseSettings):
    """Speech-to-text (Moonshine v2 via moonshine-voice) configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_STT_")

    # moonshine-voice downloads to its own cache dir;
    # path and arch are discovered via `python -m moonshine_voice.download`
    model_path: str = ""  # empty = auto-discover from moonshine cache
    model_arch: int = 5  # 5 = medium-streaming-en quantized


class TTSSettings(BaseSettings):
    """Text-to-speech (Kokoro-82M ONNX) configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_TTS_")

    model_path: Path = Path("./data/models/kokoro/kokoro-v1.0.int8.onnx")
    voices_path: Path = Path("./data/models/kokoro/voices-v1.0.bin")
    voice: str = "af_bella"
    speed: float = 1.0
    sample_rate: int = 24000
    lang: str = "en-us"


class VADSettings(BaseSettings):
    """Voice activity detection (Silero VAD) configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_VAD_")

    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 800
    window_size_samples: int = 512  # silero expects 512 samples at 16kHz (32ms)


class WakeWordSettings(BaseSettings):
    """Wake word detection (openWakeWord) configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_WAKE_WORD_")

    # Uses pre-trained "hey_jarvis" until custom "hey_ara" is trained
    model_names: list[str] = ["hey_jarvis_v0.1"]
    threshold: float = 0.5
    enabled: bool = True
    inference_framework: str = "onnx"


class MemorySettings(BaseSettings):
    """Memory, RAG, and knowledge graph configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_")

    vector_db_path: Path = Path("./data/vector_db")
    graph_db_path: Path = Path("./data/graph_db/graph.sqlite")
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 20
    rerank_top_k: int = 5


class ConversationSettings(BaseSettings):
    """Conversation persistence and memory feedback loop configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_CONVERSATION_")

    silence_timeout_s: int = 300
    min_turns_to_persist: int = 3

    # This uses a different prefix pattern, so set via ARA_ prefix in parent
    db_path: Path = Path("./data/conversations.sqlite")


class VisionSettings(BaseSettings):
    """Vision pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_")

    moondream_model: str = "vikhyatk/moondream2"
    qwen_vl_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vision_always_on: bool = True


class InternetSettings(BaseSettings):
    """Internet connectivity layer configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_")

    internet_enabled: bool = True
    searxng_url: str = "http://localhost:8888"
    web_cache_ttl_s: int = 3600
    max_research_steps: int = 5


class PersonalitySettings(BaseSettings):
    """ARA personality and humor configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_")

    humor_dial: int = Field(default=7, ge=1, le=10)
    voice_identity: Literal["female"] = "female"
    auto_dial_down_on_distress: bool = True
    distress_dial_reduction: int = Field(default=2, ge=0, le=5)


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_API_")

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]


class DesktopSettings(BaseSettings):
    """Desktop automation configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_DESKTOP_")

    enabled: bool = True
    require_confirmation: bool = True


class AraSettings(BaseSettings):
    """Root settings — aggregates all sub-settings for the ARA application.

    Usage:
        settings = AraSettings()
        print(settings.llm.primary_model)
        print(settings.personality.humor_dial)
    """

    model_config = SettingsConfigDict(
        env_prefix="ARA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    env: Literal["development", "production", "testing"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    data_dir: Path = Path("./data")

    # Hotkey
    hotkey: str = "ctrl+space"

    # Sub-settings (loaded independently from env vars)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    wake_word: WakeWordSettings = Field(default_factory=WakeWordSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    internet: InternetSettings = Field(default_factory=InternetSettings)
    personality: PersonalitySettings = Field(default_factory=PersonalitySettings)
    api: APISettings = Field(default_factory=APISettings)
    desktop: DesktopSettings = Field(default_factory=DesktopSettings)

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: str | Path) -> Path:
        return Path(v).resolve()


def get_settings() -> AraSettings:
    """Create and return application settings. Reads from .env file and env vars."""
    return AraSettings()
