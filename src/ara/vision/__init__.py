"""Vision pipeline — Moondream (always-on), Qwen-VL (on-demand), image analysis.

Exports:
    Models:
        MoondreamModel — Always-on lightweight vision model (2.5GB VRAM)
        QwenVLModel — On-demand complex vision via Ollama (5GB VRAM)

    Pipeline:
        VisionPipeline — Orchestrator that routes tasks and applies personality
        PipelineResult — Result combining raw output + personality text

    Types:
        VisionResult — Raw model output with timing and metadata
        VisionTask — Simple task type enum (CAPTION, VQA, DETECT)
        VisionTaskType — Full task type enum including complex tasks
        VisionComplexity — SIMPLE vs COMPLEX routing classification
"""

from ara.vision.moondream import MoondreamModel, VisionResult, VisionTask
from ara.vision.pipeline import (
    PipelineResult,
    VisionComplexity,
    VisionPipeline,
    VisionTaskType,
)
from ara.vision.qwen_vl import QwenVLModel

__all__ = [
    "MoondreamModel",
    "PipelineResult",
    "QwenVLModel",
    "VisionComplexity",
    "VisionPipeline",
    "VisionResult",
    "VisionTask",
    "VisionTaskType",
]
