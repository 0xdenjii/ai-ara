"""Vision pipeline — orchestrates Moondream (always-on) and Qwen-VL (on-demand).

Provides:
    - VisionPipeline: Unified API for all vision tasks (caption, VQA, OCR, etc.)
    - Model routing: simple tasks → Moondream, complex tasks → Qwen-VL
    - Personality injection: raw model output → ARA-style descriptions via LLM
    - Memory integration: stores visual context as embeddings in LanceDB

Architecture:
    The pipeline decides which model to use based on task complexity:
        - CAPTION, VQA, DETECT → Moondream (2.5GB, always loaded)
        - OCR, DOCUMENT, DETAILED → Qwen-VL (5GB, loaded on demand via Ollama)

    For personality injection, the pipeline sends the raw model output through
    the primary LLM with a system prompt instructing ARA-style reformulation.

    Memory integration stores image descriptions as MemoryChunks in LanceDB
    with source=VISION so they can be recalled in future conversations.

Usage:
    pipeline = VisionPipeline(settings)
    await pipeline.start()
    result = await pipeline.describe(image)
    result = await pipeline.ask(image, "What color is the car?")
    result = await pipeline.ocr(image)
    await pipeline.stop()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ara.utils.errors import VisionError
from ara.utils.logging import get_logger
from ara.vision.moondream import MoondreamModel, VisionResult
from ara.vision.qwen_vl import QwenVLModel

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient

log = get_logger("vision_pipeline")


# ── Task Complexity Routing ──────────────────────────────────────────────────


class VisionComplexity(StrEnum):
    """Complexity level determines which model handles the task."""

    SIMPLE = "simple"  # Moondream: caption, basic VQA, detection
    COMPLEX = "complex"  # Qwen-VL: OCR, document analysis, detailed reasoning


class VisionTaskType(StrEnum):
    """High-level task types the pipeline supports."""

    CAPTION = "caption"
    VQA = "vqa"
    DETECT = "detect"
    OCR = "ocr"
    DOCUMENT = "document"
    DETAILED = "detailed"


# Map task types to complexity (and thus to model choice)
_TASK_COMPLEXITY: dict[VisionTaskType, VisionComplexity] = {
    VisionTaskType.CAPTION: VisionComplexity.SIMPLE,
    VisionTaskType.VQA: VisionComplexity.SIMPLE,
    VisionTaskType.DETECT: VisionComplexity.SIMPLE,
    VisionTaskType.OCR: VisionComplexity.COMPLEX,
    VisionTaskType.DOCUMENT: VisionComplexity.COMPLEX,
    VisionTaskType.DETAILED: VisionComplexity.COMPLEX,
}


# ── Pipeline Result ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineResult:
    """Result from the vision pipeline, combining model output with personality.

    Attributes:
        raw_result: The raw VisionResult from the underlying model.
        personality_text: ARA-style reformulated description (if personality enabled).
        task_type: What type of analysis was performed.
        model_used: Which model handled this ("moondream" or "qwen_vl").
        total_time_ms: Total pipeline time including personality injection.
        stored_in_memory: Whether the result was saved to vector store.
    """

    raw_result: VisionResult
    personality_text: str
    task_type: VisionTaskType
    model_used: str
    total_time_ms: float
    stored_in_memory: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Best available text — personality version if available, else raw."""
        return self.personality_text if self.personality_text else self.raw_result.text


# ── Personality Prompts ──────────────────────────────────────────────────────

_PERSONALITY_SYSTEM = """You are ARA — a sharp, witty personal AI best friend describing what you see.

Rules:
- Be conversational, natural, brief. Sound like a real person talking to a friend.
- No corporate speak. No "The image depicts..." or "I can observe that..."
- Use contractions, casual language, fragments.
- If something is funny or notable in the image, comment on it.
- Keep it to 1-3 sentences unless the user asked for detail.
- Don't describe obvious things like "there is a background" — focus on what matters.
- Match the energy: mundane images get quick descriptions, interesting ones get more.
- Never say "As an AI..." or break the friend illusion."""

_PERSONALITY_CAPTION_PROMPT = """Rewrite this image description in your voice (ARA-style, casual, brief):

Raw description: {raw_text}

Your version:"""

_PERSONALITY_VQA_PROMPT = """The user asked about an image: "{question}"

The analysis found: {raw_text}

Answer the user naturally in your voice (casual, direct, ARA-style):"""

_PERSONALITY_OCR_PROMPT = """I extracted text from an image. Present it cleanly to the user:

Extracted text: {raw_text}

Present this naturally — keep the actual text intact but add a brief intro in your voice:"""


# ── Vision Pipeline ──────────────────────────────────────────────────────────


class VisionPipeline:
    """Orchestrates vision models with personality injection and memory storage.

    Routes tasks to Moondream (simple) or Qwen-VL (complex), optionally
    reformulates descriptions through the primary LLM for ARA's personality,
    and can store visual context in the vector store for future recall.

    Args:
        settings: Application settings.
        llm_client: Optional LLM client for personality injection.
            If not provided, raw model output is returned without reformulation.
        memory_store_fn: Optional async callback to store vision results in memory.
            Signature: async (text: str, source_file: str, metadata: dict) -> None
    """

    def __init__(
        self,
        settings: AraSettings,
        *,
        llm_client: LLMClient | None = None,
        memory_store_fn: Any | None = None,
    ) -> None:
        self._settings = settings
        self._llm_client = llm_client
        self._memory_store_fn = memory_store_fn
        self._personality_enabled: bool = llm_client is not None

        # Model instances
        self._moondream = MoondreamModel(settings)
        self._qwen_vl = QwenVLModel(
            settings,
            ollama_model=settings.vision.qwen_vl_model
            if ":" in settings.vision.qwen_vl_model
            else "qwen2.5-vl:7b",
        )

        self._started: bool = False

    @property
    def started(self) -> bool:
        """Whether the pipeline has been initialized."""
        return self._started

    @property
    def moondream_loaded(self) -> bool:
        """Whether Moondream is currently loaded."""
        return self._moondream.loaded

    @property
    def qwen_vl_loaded(self) -> bool:
        """Whether Qwen-VL is available in Ollama."""
        return self._qwen_vl.loaded

    @property
    def personality_enabled(self) -> bool:
        """Whether personality injection is active."""
        return self._personality_enabled

    async def start(self) -> None:
        """Initialize the pipeline — load Moondream (always-on model).

        Qwen-VL is loaded on-demand when needed for complex tasks.

        Raises:
            VisionError: If Moondream fails to load.
        """
        if self._started:
            return

        try:
            if self._settings.vision.vision_always_on:
                log.info("vision_pipeline_loading_moondream")
                await self._moondream.load()
            else:
                log.info("vision_pipeline_moondream_disabled", reason="vision_always_on=False")

            self._started = True
            log.info(
                "vision_pipeline_started",
                moondream_loaded=self._moondream.loaded,
                personality=self._personality_enabled,
            )
        except Exception as exc:
            raise VisionError(
                f"Failed to start vision pipeline: {exc}",
                operation="start",
            ) from exc

    async def stop(self) -> None:
        """Shut down the pipeline — unload all models."""
        if not self._started:
            return

        await self._moondream.unload()
        await self._qwen_vl.unload()

        self._started = False
        log.info("vision_pipeline_stopped")

    def _ensure_started(self) -> None:
        """Raise if pipeline is not started."""
        if not self._started:
            raise VisionError(
                "Vision pipeline not started. Call start() first.",
                operation="check",
            )

    def _get_complexity(self, task_type: VisionTaskType) -> VisionComplexity:
        """Determine which model should handle this task."""
        return _TASK_COMPLEXITY.get(task_type, VisionComplexity.SIMPLE)

    async def _ensure_qwen_vl(self) -> None:
        """Load Qwen-VL on demand if not already available.

        Raises:
            VisionError: If loading fails.
        """
        if not self._qwen_vl.loaded:
            log.info("vision_pipeline_loading_qwen_vl_on_demand")
            await self._qwen_vl.load()

    async def _ensure_moondream(self) -> None:
        """Load Moondream if not already loaded.

        Raises:
            VisionError: If loading fails.
        """
        if not self._moondream.loaded:
            log.info("vision_pipeline_loading_moondream_on_demand")
            await self._moondream.load()

    async def _apply_personality(
        self,
        raw_text: str,
        task_type: VisionTaskType,
        question: str = "",
    ) -> str:
        """Reformulate raw model output through the LLM with ARA's personality.

        Args:
            raw_text: Raw description from the vision model.
            task_type: What kind of task produced this text.
            question: Original user question (for VQA).

        Returns:
            ARA-style reformulated text, or raw text if personality is disabled.
        """
        if not self._personality_enabled or self._llm_client is None:
            return raw_text

        if not raw_text.strip():
            return raw_text

        # Select the appropriate prompt template
        if task_type in (VisionTaskType.OCR, VisionTaskType.DOCUMENT):
            user_prompt = _PERSONALITY_OCR_PROMPT.format(raw_text=raw_text)
        elif task_type == VisionTaskType.VQA and question:
            user_prompt = _PERSONALITY_VQA_PROMPT.format(
                question=question,
                raw_text=raw_text,
            )
        else:
            user_prompt = _PERSONALITY_CAPTION_PROMPT.format(raw_text=raw_text)

        try:
            from ara.utils.types import ChatMessage, MessageRole

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=_PERSONALITY_SYSTEM),
                ChatMessage(role=MessageRole.USER, content=user_prompt),
            ]
            personality_text = await self._llm_client.generate(
                messages,
                temperature=0.8,
                max_tokens=512,
            )
            return personality_text.strip()
        except Exception as exc:
            log.warning(
                "personality_injection_failed",
                error=str(exc),
                fallback="raw_text",
            )
            return raw_text

    async def _store_in_memory(
        self,
        text: str,
        task_type: VisionTaskType,
        image_path: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store vision result in memory via the callback.

        Args:
            text: Description text to store.
            task_type: Task type for metadata.
            image_path: Source image path (if available).
            extra_metadata: Additional metadata to include.

        Returns:
            True if stored successfully, False otherwise.
        """
        if self._memory_store_fn is None:
            return False

        metadata = {
            "task_type": str(task_type),
            "source": "vision",
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        try:
            await self._memory_store_fn(
                text=text,
                source_file=image_path or f"vision_{task_type}",
                metadata=metadata,
            )
            log.info("vision_result_stored", task_type=str(task_type))
            return True
        except Exception as exc:
            log.warning("vision_memory_store_failed", error=str(exc))
            return False

    async def _run_simple_task(
        self,
        image: Image.Image,
        task_type: VisionTaskType,
        question: str = "",
        target: str = "",
    ) -> VisionResult:
        """Execute a simple task via Moondream.

        Args:
            image: PIL Image to analyze.
            task_type: Type of simple task.
            question: Question text (for VQA).
            target: Detection target (for DETECT).

        Returns:
            VisionResult from Moondream.
        """
        await self._ensure_moondream()

        if task_type == VisionTaskType.CAPTION:
            return await self._moondream.caption(image)

        if task_type == VisionTaskType.VQA:
            if not question:
                raise VisionError(
                    "VQA requires a question",
                    operation="vqa",
                )
            return await self._moondream.ask(image, question)

        if task_type == VisionTaskType.DETECT:
            if not target:
                raise VisionError(
                    "Detection requires a target description",
                    operation="detect",
                )
            return await self._moondream.detect(image, target)

        # Fallback to caption for unknown simple tasks
        return await self._moondream.caption(image)

    async def _run_complex_task(
        self,
        image: Image.Image,
        task_type: VisionTaskType,
        prompt: str = "",
    ) -> VisionResult:
        """Execute a complex task via Qwen-VL.

        Args:
            image: PIL Image to analyze.
            task_type: Type of complex task.
            prompt: Custom prompt (for DETAILED).

        Returns:
            VisionResult from Qwen-VL.
        """
        await self._ensure_qwen_vl()

        if task_type == VisionTaskType.OCR:
            return await self._qwen_vl.ocr(image)

        if task_type == VisionTaskType.DOCUMENT:
            return await self._qwen_vl.analyze_document(image)

        if task_type == VisionTaskType.DETAILED:
            custom_prompt = prompt or "Describe this image in thorough detail."
            return await self._qwen_vl.analyze(image, custom_prompt)

        return await self._qwen_vl.analyze(image)

    async def analyze(
        self,
        image: Image.Image,
        task_type: VisionTaskType = VisionTaskType.CAPTION,
        *,
        question: str = "",
        target: str = "",
        prompt: str = "",
        personality: bool = True,
        store_memory: bool = False,
        image_path: str = "",
    ) -> PipelineResult:
        """Unified vision analysis entry point.

        Routes to the appropriate model based on task complexity, optionally
        applies personality, and stores in memory.

        Args:
            image: PIL Image to analyze.
            task_type: What kind of analysis to perform.
            question: Question text (required for VQA).
            target: Detection target (required for DETECT).
            prompt: Custom prompt (for DETAILED tasks).
            personality: Whether to apply ARA personality to output.
            store_memory: Whether to save result to vector store.
            image_path: Source image path (for memory metadata).

        Returns:
            PipelineResult with raw and personality-infused text.
        """
        self._ensure_started()
        pipeline_start = time.monotonic()

        complexity = self._get_complexity(task_type)

        # Route to appropriate model
        if complexity == VisionComplexity.SIMPLE:
            raw_result = await self._run_simple_task(image, task_type, question, target)
            model_used = "moondream"
        else:
            raw_result = await self._run_complex_task(image, task_type, prompt)
            model_used = "qwen_vl"

        # Apply personality
        personality_text = ""
        if personality and self._personality_enabled:
            personality_text = await self._apply_personality(
                raw_result.text,
                task_type,
                question,
            )

        # Store in memory
        stored = False
        if store_memory:
            text_to_store = personality_text if personality_text else raw_result.text
            stored = await self._store_in_memory(
                text_to_store,
                task_type,
                image_path,
                extra_metadata={
                    "model": model_used,
                    "image_size": f"{raw_result.image_size[0]}x{raw_result.image_size[1]}",
                    "inference_ms": raw_result.inference_time_ms,
                },
            )

        total_ms = (time.monotonic() - pipeline_start) * 1000

        return PipelineResult(
            raw_result=raw_result,
            personality_text=personality_text,
            task_type=task_type,
            model_used=model_used,
            total_time_ms=round(total_ms, 2),
            stored_in_memory=stored,
            metadata={
                "complexity": str(complexity),
                "personality_applied": bool(personality_text),
            },
        )

    # ── Convenience Methods ──────────────────────────────────────────────────

    async def describe(
        self,
        image: Image.Image,
        *,
        personality: bool = True,
        store_memory: bool = False,
        image_path: str = "",
    ) -> PipelineResult:
        """Generate a caption/description for an image.

        Uses Moondream for quick descriptions.
        """
        return await self.analyze(
            image,
            VisionTaskType.CAPTION,
            personality=personality,
            store_memory=store_memory,
            image_path=image_path,
        )

    async def ask(
        self,
        image: Image.Image,
        question: str,
        *,
        personality: bool = True,
    ) -> PipelineResult:
        """Answer a question about an image (VQA).

        Uses Moondream for quick question answering.
        """
        return await self.analyze(
            image,
            VisionTaskType.VQA,
            question=question,
            personality=personality,
        )

    async def detect(
        self,
        image: Image.Image,
        target: str,
        *,
        personality: bool = True,
    ) -> PipelineResult:
        """Detect objects matching a description.

        Uses Moondream for detection.
        """
        return await self.analyze(
            image,
            VisionTaskType.DETECT,
            target=target,
            personality=personality,
        )

    async def ocr(
        self,
        image: Image.Image,
        *,
        personality: bool = True,
        store_memory: bool = False,
        image_path: str = "",
    ) -> PipelineResult:
        """Extract text from an image via OCR.

        Uses Qwen-VL for accurate text recognition.
        """
        return await self.analyze(
            image,
            VisionTaskType.OCR,
            personality=personality,
            store_memory=store_memory,
            image_path=image_path,
        )

    async def analyze_document(
        self,
        image: Image.Image,
        *,
        personality: bool = True,
        store_memory: bool = False,
        image_path: str = "",
    ) -> PipelineResult:
        """Analyze a document image (receipt, form, screenshot, etc.).

        Uses Qwen-VL for document understanding.
        """
        return await self.analyze(
            image,
            VisionTaskType.DOCUMENT,
            personality=personality,
            store_memory=store_memory,
            image_path=image_path,
        )

    async def detailed_analysis(
        self,
        image: Image.Image,
        prompt: str = "",
        *,
        personality: bool = True,
        store_memory: bool = False,
        image_path: str = "",
    ) -> PipelineResult:
        """Perform detailed analysis with a custom prompt.

        Uses Qwen-VL for complex visual reasoning.
        """
        return await self.analyze(
            image,
            VisionTaskType.DETAILED,
            prompt=prompt,
            personality=personality,
            store_memory=store_memory,
            image_path=image_path,
        )

    async def describe_from_path(
        self,
        image_path: Path,
        *,
        personality: bool = True,
        store_memory: bool = False,
    ) -> PipelineResult:
        """Describe an image from a file path.

        Convenience method that opens the image and calls describe().

        Args:
            image_path: Path to the image file.
            personality: Whether to apply personality.
            store_memory: Whether to save to vector store.

        Returns:
            PipelineResult with description.
        """
        from PIL import Image as PILImage

        if not image_path.exists():
            raise VisionError(
                f"Image file not found: {image_path}",
                operation="describe",
                details={"path": str(image_path)},
            )

        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as exc:
            raise VisionError(
                f"Failed to open image: {exc}",
                operation="describe",
                details={"path": str(image_path)},
            ) from exc

        return await self.describe(
            image,
            personality=personality,
            store_memory=store_memory,
            image_path=str(image_path),
        )

    def get_stats(self) -> dict[str, Any]:
        """Return pipeline status and model info."""
        return {
            "started": self._started,
            "personality_enabled": self._personality_enabled,
            "memory_enabled": self._memory_store_fn is not None,
            "moondream": self._moondream.get_stats(),
            "qwen_vl": self._qwen_vl.get_stats(),
        }
