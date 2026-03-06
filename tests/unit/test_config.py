"""Tests for ARA configuration system."""

from __future__ import annotations

from ara.config.models import (
    MODELS,
    TOTAL_VRAM_GB,
    ModelLocation,
    ModelRole,
    calculate_vram_usage,
    get_daily_mode_budget,
    get_long_context_mode_budget,
    get_total_cpu_ram_mb,
)
from ara.config.settings import AraSettings


class TestAraSettings:
    """Tests for the root settings object."""

    def test_default_settings_load(self) -> None:
        settings = AraSettings()
        assert settings.env == "development"
        assert settings.log_level == "INFO"

    def test_llm_defaults(self) -> None:
        settings = AraSettings()
        assert settings.llm.primary_model == "qwen3.5:35b"
        assert settings.llm.temperature == 0.7
        assert settings.llm.max_response_tokens == 4096

    def test_voice_defaults(self) -> None:
        settings = AraSettings()
        assert settings.audio.sample_rate == 16000
        assert settings.stt.model_arch == 5
        assert settings.tts.voice == "af_bella"
        assert settings.tts.lang == "en-us"
        assert settings.vad.threshold == 0.5
        assert settings.wake_word.enabled is True
        assert "hey_jarvis_v0.1" in settings.wake_word.model_names

    def test_memory_defaults(self) -> None:
        settings = AraSettings()
        assert settings.memory.chunk_size == 512
        assert settings.memory.chunk_overlap == 64
        assert settings.memory.retrieval_top_k == 20
        assert settings.memory.rerank_top_k == 5

    def test_personality_defaults(self) -> None:
        settings = AraSettings()
        assert settings.personality.humor_dial == 7
        assert settings.personality.voice_identity == "female"
        assert settings.personality.auto_dial_down_on_distress is True
        assert settings.personality.distress_dial_reduction == 2

    def test_internet_defaults(self) -> None:
        settings = AraSettings()
        assert settings.internet.internet_enabled is True
        assert settings.internet.max_research_steps == 5

    def test_api_defaults(self) -> None:
        settings = AraSettings()
        assert settings.api.host == "127.0.0.1"
        assert settings.api.port == 8000

    def test_data_dir_resolves_to_absolute(self) -> None:
        settings = AraSettings()
        assert settings.data_dir.is_absolute()

    def test_conversation_defaults(self) -> None:
        settings = AraSettings()
        assert settings.conversation.silence_timeout_s == 300
        assert settings.conversation.min_turns_to_persist == 3


class TestModelRegistry:
    """Tests for the model registry and VRAM budget calculator."""

    def test_all_roles_have_specs(self) -> None:
        for role in ModelRole:
            assert role in MODELS, f"Missing model spec for {role}"

    def test_gpu_models_have_vram(self) -> None:
        for role, spec in MODELS.items():
            if spec.location == ModelLocation.GPU:
                # VISION_ON_DEMAND uses native vision in primary LLM — no separate VRAM
                if role == ModelRole.VISION_ON_DEMAND:
                    assert spec.vram_gb >= 0, f"{role} is GPU but has negative VRAM"
                else:
                    assert spec.vram_gb > 0, f"{role} is GPU but has 0 VRAM"

    def test_cpu_models_have_zero_vram(self) -> None:
        for role, spec in MODELS.items():
            if spec.location == ModelLocation.CPU:
                assert spec.vram_gb == 0, f"{role} is CPU but has non-zero VRAM"

    def test_daily_mode_fits_in_vram(self) -> None:
        budget = get_daily_mode_budget()
        assert budget["total"] <= TOTAL_VRAM_GB
        assert budget["headroom"] > 0

    def test_long_context_mode_fits_in_vram(self) -> None:
        budget = get_long_context_mode_budget()
        assert budget["total"] <= TOTAL_VRAM_GB
        assert budget["headroom"] > 0

    def test_calculate_vram_returns_headroom(self) -> None:
        usage = calculate_vram_usage([ModelRole.PRIMARY_LLM])
        assert "total" in usage
        assert "headroom" in usage
        assert usage["headroom"] == TOTAL_VRAM_GB - usage["total"]

    def test_total_cpu_ram_is_reasonable(self) -> None:
        total_mb = get_total_cpu_ram_mb()
        # All CPU models should be under 4GB total
        assert 0 < total_mb < 4096
