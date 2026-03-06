"""Model registry — paths, VRAM budgets, and metadata for all ML models used by ARA.

This module provides a single source of truth for model specifications.
No hardcoded model paths or VRAM numbers elsewhere in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelLocation(StrEnum):
    """Where a model runs."""

    GPU = "gpu"
    CPU = "cpu"


class ModelRole(StrEnum):
    """Functional role of a model in the system."""

    PRIMARY_LLM = "primary_llm"
    LONG_CONTEXT_LLM = "long_context_llm"
    DRAFT_LLM = "draft_llm"
    STT = "stt"
    TTS = "tts"
    VAD = "vad"
    WAKE_WORD = "wake_word"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    VISION_ALWAYS_ON = "vision_always_on"
    VISION_ON_DEMAND = "vision_on_demand"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single ML model."""

    role: ModelRole
    name: str
    location: ModelLocation
    vram_gb: float
    ram_mb: float
    description: str
    quantization: str | None = None
    context_length: int | None = None
    license: str | None = None


# ── Model Registry ───────────────────────────────────────────────────────────
# All models ARA uses, with their resource requirements.

MODELS: dict[ModelRole, ModelSpec] = {
    ModelRole.PRIMARY_LLM: ModelSpec(
        role=ModelRole.PRIMARY_LLM,
        name="Qwen3.5-35B-A3B",
        location=ModelLocation.GPU,
        vram_gb=20.5,
        ram_mb=0,
        description=(
            "Hybrid DeltaNet+MoE: 35B total, 3B active per token, 256 experts. "
            "~122 t/s @ 32K. Native multimodal (text+image+video). 256K context. "
            "Uses q8_0 KV cache + flash attention for comfortable 24GB VRAM fit."
        ),
        quantization="Q4_K_M",
        context_length=262144,
        license="Apache-2.0",
    ),
    ModelRole.LONG_CONTEXT_LLM: ModelSpec(
        role=ModelRole.LONG_CONTEXT_LLM,
        name="Qwen3.5-27B",
        location=ModelLocation.GPU,
        vram_gb=17.0,
        ram_mb=0,
        description=(
            "Dense 27B for deep analysis. All params active per token. "
            "262K native context. ~35-45 t/s. Hot-swap with primary."
        ),
        quantization="Q4_K_M",
        context_length=262144,
        license="Apache-2.0",
    ),
    ModelRole.DRAFT_LLM: ModelSpec(
        role=ModelRole.DRAFT_LLM,
        name="Qwen3-1B",
        location=ModelLocation.GPU,
        vram_gb=1.0,
        ram_mb=0,
        description="Draft model for speculative decoding. Predicts tokens for main model to verify.",
        quantization="Q4_0",
        context_length=32768,
        license="Apache-2.0",
    ),
    ModelRole.STT: ModelSpec(
        role=ModelRole.STT,
        name="Moonshine-v2-Medium",
        location=ModelLocation.CPU,
        vram_gb=0,
        ram_mb=300,
        description="Streaming STT, ~50ms residual latency, English-only, ONNX.",
        license="MIT",
    ),
    ModelRole.TTS: ModelSpec(
        role=ModelRole.TTS,
        name="Kokoro-82M",
        location=ModelLocation.CPU,
        vram_gb=0,
        ram_mb=200,
        description="Sub-300ms, #1 TTS Arena, 50+ voice packs, ONNX.",
        license="Apache-2.0",
    ),
    ModelRole.VAD: ModelSpec(
        role=ModelRole.VAD,
        name="Silero-VAD",
        location=ModelLocation.CPU,
        vram_gb=0,
        ram_mb=2,
        description="Voice activity detection, ~1ms per chunk.",
        license="MIT",
    ),
    ModelRole.WAKE_WORD: ModelSpec(
        role=ModelRole.WAKE_WORD,
        name="openWakeWord",
        location=ModelLocation.CPU,
        vram_gb=0,
        ram_mb=10,
        description='Custom "Hey Ara" wake word detection.',
        license="Apache-2.0",
    ),
    ModelRole.EMBEDDING: ModelSpec(
        role=ModelRole.EMBEDDING,
        name="nomic-embed-text-v1.5",
        location=ModelLocation.CPU,
        vram_gb=0,
        ram_mb=262,
        description="137M params, 768-dim embeddings for vector search.",
        license="Apache-2.0",
    ),
    ModelRole.RERANKER: ModelSpec(
        role=ModelRole.RERANKER,
        name="bge-reranker-v2-m3",
        location=ModelLocation.CPU,
        vram_gb=0,
        ram_mb=1100,
        description="Cross-encoder reranker, +8-11% retrieval accuracy.",
        license="Apache-2.0",
    ),
    ModelRole.VISION_ALWAYS_ON: ModelSpec(
        role=ModelRole.VISION_ALWAYS_ON,
        name="Moondream-2B",
        location=ModelLocation.GPU,
        vram_gb=2.5,
        ram_mb=0,
        description="Quick image descriptions, always loaded alongside primary LLM.",
        quantization="4-bit",
        license="Apache-2.0",
    ),
    ModelRole.VISION_ON_DEMAND: ModelSpec(
        role=ModelRole.VISION_ON_DEMAND,
        name="Qwen3.5-35B-A3B-Vision",
        location=ModelLocation.GPU,
        vram_gb=0,
        ram_mb=0,
        description=(
            "Native vision via Qwen3.5-35B-A3B. No separate model needed — "
            "primary LLM includes integrated SigLIP vision encoder for OCR, "
            "document analysis, and complex image understanding."
        ),
        quantization="Q4_K_M",
        license="Apache-2.0",
    ),
}

# ── VRAM Budget ──────────────────────────────────────────────────────────────

TOTAL_VRAM_GB: float = 24.0


def calculate_vram_usage(active_roles: list[ModelRole]) -> dict[str, float]:
    """Calculate VRAM usage for a set of active model roles.

    Returns dict with per-model usage and total, plus headroom.
    """
    usage: dict[str, float] = {}
    total = 0.0

    for role in active_roles:
        spec = MODELS[role]
        if spec.location == ModelLocation.GPU:
            usage[spec.name] = spec.vram_gb
            total += spec.vram_gb

    usage["total"] = total
    usage["headroom"] = TOTAL_VRAM_GB - total
    return usage


def get_daily_mode_budget() -> dict[str, float]:
    """VRAM budget for daily conversation mode."""
    return calculate_vram_usage(
        [
            ModelRole.PRIMARY_LLM,
            ModelRole.VISION_ALWAYS_ON,
        ]
    )


def get_long_context_mode_budget() -> dict[str, float]:
    """VRAM budget for long-context diary analysis mode."""
    return calculate_vram_usage([ModelRole.LONG_CONTEXT_LLM])


def get_total_cpu_ram_mb() -> float:
    """Total CPU RAM used by all CPU-resident models."""
    return sum(spec.ram_mb for spec in MODELS.values() if spec.location == ModelLocation.CPU)
