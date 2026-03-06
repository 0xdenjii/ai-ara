"""ARA application settings loaded from environment variables and .env files.

All configuration is centralized here. No hardcoded values anywhere in the codebase.
Uses pydantic-settings to validate and type-check all config at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE_OPTS = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class LLMSettings(BaseSettings):
    """LLM inference configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_", **_ENV_FILE_OPTS)

    backend_type: str = "ollama"  # "ollama" or "llamacpp"
    ollama_base_url: str = "http://localhost:11434"
    llamacpp_base_url: str = "http://localhost:8080"
    primary_model: str = "qwen3.5:35b"
    long_context_model: str = "qwen3.5:27b"
    draft_model: str = "qwen3:1b"
    primary_model_context_length: int = 131072
    long_context_length: int = 262144
    temperature: float = 0.7
    max_response_tokens: int = 4096

    # ── Ollama performance tuning ────────────────────────────────────────
    # These map to Ollama server env vars (OLLAMA_FLASH_ATTENTION, etc.)
    ollama_flash_attention: bool = True  # Reduce peak VRAM during attention
    ollama_kv_cache_type: str = "q8_0"  # KV cache quantization: f16, q8_0, q4_0
    ollama_num_ctx: int = 131072  # Context length for Ollama requests

    # ── Speculative Decoding (llama.cpp server only) ─────────────────────
    # These settings configure llama-server's speculative decoding feature.
    # The draft model generates candidate tokens that the primary model
    # verifies in parallel, typically achieving 1.5-2.5x speedup.
    speculative_enabled: bool = False
    speculative_draft_max: int = 16  # Max draft tokens per speculation step
    speculative_draft_min: int = 5  # Min draft tokens to trigger speculation
    speculative_draft_p_min: float = 0.9  # Min probability for draft acceptance
    speculative_n_gpu_layers_draft: int = -1  # GPU layers for draft model (-1 = all)

    # ── llama.cpp Server Options ─────────────────────────────────────────
    llamacpp_n_gpu_layers: int = -1  # GPU layers for primary model (-1 = all)
    llamacpp_flash_attn: bool = True  # Enable flash attention (recommended for RTX 4090)
    llamacpp_kv_cache_type_k: str = "q8_0"  # KV cache quantization for K: f16, q8_0, q4_0
    llamacpp_kv_cache_type_v: str = "q8_0"  # KV cache quantization for V: f16, q8_0, q4_0
    llamacpp_context_size: int = 131072  # KV cache context size
    llamacpp_batch_size: int = 2048  # Batch size for prompt processing
    llamacpp_threads: int = 0  # CPU threads (0 = auto-detect)
    llamacpp_host: str = "127.0.0.1"
    llamacpp_port: int = 8080
    llamacpp_generation_timeout_s: float = 300.0  # Timeout for generation requests (5 min)
    llamacpp_health_timeout_s: float = 10.0  # Timeout for health/connectivity checks


class AudioSettings(BaseSettings):
    """Audio capture and playback configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_AUDIO_", **_ENV_FILE_OPTS)

    sample_rate: int = 16000
    channels: int = 1
    block_duration_ms: int = 32  # 32ms = 512 samples at 16kHz (matches Silero VAD window)


class STTSettings(BaseSettings):
    """Speech-to-text (Moonshine v2 via moonshine-voice) configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_STT_", **_ENV_FILE_OPTS)

    # moonshine-voice downloads to its own cache dir;
    # path and arch are discovered via `python -m moonshine_voice.download`
    model_path: str = ""  # empty = auto-discover from moonshine cache
    model_arch: int = 5  # 5 = medium-streaming-en quantized


class TTSSettings(BaseSettings):
    """Text-to-speech (Kokoro-82M ONNX) configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_TTS_", **_ENV_FILE_OPTS)

    model_path: Path = Path("./data/models/kokoro/kokoro-v1.0.int8.onnx")
    voices_path: Path = Path("./data/models/kokoro/voices-v1.0.bin")
    voice: str = "af_bella"
    speed: float = 1.0
    sample_rate: int = 24000
    lang: str = "en-us"


class VADSettings(BaseSettings):
    """Voice activity detection (Silero VAD) configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_VAD_", **_ENV_FILE_OPTS)

    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 800
    window_size_samples: int = 512  # silero expects 512 samples at 16kHz (32ms)


class WakeWordSettings(BaseSettings):
    """Wake word detection (openWakeWord) configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_WAKE_WORD_", **_ENV_FILE_OPTS)

    # Uses pre-trained "hey_jarvis" until custom "hey_ara" is trained
    model_names: list[str] = ["hey_jarvis_v0.1"]
    threshold: float = 0.5
    enabled: bool = True
    inference_framework: str = "onnx"


class MemorySettings(BaseSettings):
    """Memory, RAG, and knowledge graph configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_", **_ENV_FILE_OPTS)

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

    model_config = SettingsConfigDict(env_prefix="ARA_CONVERSATION_", **_ENV_FILE_OPTS)

    silence_timeout_s: int = 300
    min_turns_to_persist: int = 3

    # This uses a different prefix pattern, so set via ARA_ prefix in parent
    db_path: Path = Path("./data/conversations.sqlite")


class VisionSettings(BaseSettings):
    """Vision pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_", **_ENV_FILE_OPTS)

    moondream_model: str = "vikhyatk/moondream2"
    qwen_vl_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vision_always_on: bool = True


class InternetSettings(BaseSettings):
    """Internet connectivity layer configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_", **_ENV_FILE_OPTS)

    internet_enabled: bool = True
    searxng_url: str = "http://localhost:8888"
    web_cache_ttl_s: int = 3600
    max_research_steps: int = 5
    connectivity_check_timeout_s: float = 3.0  # Timeout for individual connectivity checks
    connectivity_check_interval_s: float = 10.0  # Min interval between network checks
    connectivity_background_interval_s: float = 60.0  # Background check interval


class PersonalitySettings(BaseSettings):
    """ARA personality and humor configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_", **_ENV_FILE_OPTS)

    humor_dial: int = Field(default=7, ge=1, le=10)
    voice_identity: Literal["female"] = "female"
    auto_dial_down_on_distress: bool = True
    distress_dial_reduction: int = Field(default=2, ge=0, le=5)

    # User preference fields for dynamic prompt personalization
    user_name: str = ""
    preferred_response_length: Literal["concise", "normal", "detailed"] = "normal"
    topics_to_avoid: list[str] = Field(default_factory=list)
    preferred_languages: list[str] = Field(default_factory=lambda: ["english"])
    user_traits: list[str] = Field(default_factory=list)


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_API_", **_ENV_FILE_OPTS)

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]


class DesktopSettings(BaseSettings):
    """Desktop automation configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_DESKTOP_", **_ENV_FILE_OPTS)

    enabled: bool = True
    require_confirmation: bool = True
    command_timeout_seconds: float = 30.0
    max_file_read_bytes: int = 100_000
    screenshot_dir: str = "screenshots"
    audit_db_name: str = "tool_audit.sqlite"


class ProactiveSettings(BaseSettings):
    """Proactive features configuration — suggestions, recaps, habits, mood, reviews."""

    model_config = SettingsConfigDict(env_prefix="ARA_PROACTIVE_", **_ENV_FILE_OPTS)

    enabled: bool = True

    # Scheduler loop timing
    loop_interval_s: int = 30  # Main scheduler check interval
    habit_scan_interval_s: int = 3600  # Minimum interval between habit scans (1 hour)

    # Idle-state suggestions (P12-01)
    idle_check_interval_s: int = 300  # check every 5 minutes
    idle_threshold_s: int = 600  # user idle for 10 min before suggesting
    max_suggestions_per_hour: int = 3

    # Daily recap (P12-02)
    daily_recap_enabled: bool = True
    daily_recap_hour: int = Field(default=22, ge=0, le=23)  # 10 PM
    daily_recap_minute: int = Field(default=0, ge=0, le=59)
    min_interactions_for_recap: int = 3  # skip recap on quiet days

    # Habit detection (P12-03)
    habit_detection_enabled: bool = True
    habit_min_occurrences: int = 3  # entity seen N times → potential habit
    habit_lookback_days: int = 30  # analyze last N days for patterns

    # Mood tracking (P12-04)
    mood_tracking_enabled: bool = True
    mood_window_days: int = 7  # rolling sentiment window

    # Memory reviews (P12-05)
    weekly_review_enabled: bool = True
    weekly_review_day: Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ] = "sunday"
    weekly_review_hour: int = Field(default=20, ge=0, le=23)  # 8 PM
    monthly_review_enabled: bool = True
    monthly_review_day: int = Field(default=1, ge=1, le=28)  # 1st of each month

    # Notification queue (P12-06)
    notification_queue_max: int = 20
    notification_ttl_hours: int = 24  # expire stale notifications
    notification_cooldown_s: int = 120  # min gap between delivered notifications


class ObsidianSettings(BaseSettings):
    """Obsidian vault integration configuration."""

    model_config = SettingsConfigDict(env_prefix="ARA_OBSIDIAN_", **_ENV_FILE_OPTS)

    enabled: bool = True
    api_url: str = "https://127.0.0.1:27124"
    api_key: str = ""  # Obsidian Local REST API key (from .env)
    verify_ssl: bool = False  # Local REST API uses self-signed cert

    # ── Vault structure ──────────────────────────────────────────────────
    daily_note_folder: str = "10_Journal/Daily"  # Relative to vault root
    daily_note_pattern: str = "{year}/{mm}-{month}/{date}.md"  # YYYY/MM-Month/YYYY-MM-DD.md
    inbox_path: str = "00_Inbox/RAM.md"  # Active todos / scratch pad
    efforts_folder: str = "20_Efforts"  # Projects folder

    # ── Sync engine ──────────────────────────────────────────────────────
    sync_interval_s: int = 300  # Check for vault changes every 5 minutes
    sync_on_boot: bool = True  # Full sync at ARA startup
    sync_lookback_days: int = 7  # How many days back to check for changes

    # ── Writer ───────────────────────────────────────────────────────────
    write_to_ara_log: bool = True  # Write to dedicated ## ARA's LOG section
    write_to_ops_log: bool = True  # Also write key items into Q callouts
    ara_log_heading: str = "## ARA's LOG"  # Section heading in daily note
    voice: Literal["first_person", "third_person", "factual"] = "first_person"

    # ── Morning briefing ─────────────────────────────────────────────────
    morning_briefing_enabled: bool = True
    morning_briefing_hour: int = Field(default=7, ge=0, le=23)
    end_of_day_recap_enabled: bool = True
    end_of_day_recap_hour: int = Field(default=23, ge=0, le=23)

    # ── Habit tracking ───────────────────────────────────────────────────
    protocol_tracking_enabled: bool = True
    protocol_streak_threshold_pct: float = 80.0  # Score above this = streak maintained

    # ── RAG ingestion ────────────────────────────────────────────────────
    ingest_on_sync: bool = True  # Auto-ingest changed notes into vector store
    ingest_batch_size: int = 10  # Notes per batch for ingestion


class MCPSettings(BaseSettings):
    """MCP (Model Context Protocol) server integration configuration.

    Defines external MCP servers that ARA can connect to for extended tool access.
    Server configs are auto-discovered from Claude Desktop, Antigravity, and manual entries.
    """

    model_config = SettingsConfigDict(env_prefix="ARA_MCP_", **_ENV_FILE_OPTS)

    enabled: bool = True  # Master toggle for MCP server connections
    auto_discover: bool = True  # Auto-discover from Claude Desktop / Antigravity configs
    connect_timeout_s: float = 30.0  # Timeout for connecting to each MCP server
    max_retries: int = 2  # Retries per server before giving up


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
    proactive: ProactiveSettings = Field(default_factory=ProactiveSettings)
    obsidian: ObsidianSettings = Field(default_factory=ObsidianSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: str | Path) -> Path:
        return Path(v).resolve()


def get_settings() -> AraSettings:
    """Create and return application settings. Reads from .env file and env vars."""
    return AraSettings()
