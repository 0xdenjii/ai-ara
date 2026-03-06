"""Moondream 2B wrapper — always-on lightweight vision model.

Provides:
    - MoondreamModel: Load/caption/VQA/detect objects using Moondream 2B
    - Runs on GPU (2.5GB VRAM, 4-bit quantized) alongside the primary LLM
    - Supports: captioning, visual question answering, object detection

Architecture:
    Moondream 2B is a 1.8B parameter vision-language model that encodes images
    and answers questions about them.  It runs permanently loaded in GPU memory
    alongside Qwen3-30B-A3B (total ~19GB of 24GB budget).

    The model uses the HuggingFace transformers library with torch for inference.
    Images are passed as PIL.Image objects; results are strings.

Usage:
    model = MoondreamModel(settings)
    await model.load()
    caption = await model.caption(image)
    answer = await model.ask(image, "What color is the car?")
    objects = await model.detect(image, "person")
    await model.unload()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ara.utils.errors import VisionError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from ara.config.settings import AraSettings

log = get_logger("moondream")


# ── Vision Task Types ────────────────────────────────────────────────────────


class VisionTask(StrEnum):
    """Types of vision tasks the model can perform."""

    CAPTION = "caption"
    VQA = "vqa"
    DETECT = "detect"


# ── Result Types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VisionResult:
    """Result of a vision analysis operation.

    Attributes:
        task: What type of analysis was performed.
        text: The text output (caption, answer, or detection summary).
        model_name: Which model produced this result.
        inference_time_ms: How long the inference took.
        image_size: Original image dimensions (width, height).
        metadata: Additional task-specific data (e.g., bounding boxes for detection).
    """

    task: VisionTask
    text: str
    model_name: str
    inference_time_ms: float
    image_size: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Moondream Model ──────────────────────────────────────────────────────────


class MoondreamModel:
    """Moondream 2B vision-language model wrapper.

    Supports captioning, visual question answering, and object detection.
    Designed to be always-loaded alongside the primary LLM (2.5GB VRAM).

    Args:
        settings: Application settings (provides model ID and config).
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False
        self._model_id: str = settings.vision.moondream_model
        self._device: str = "cpu"  # Determined at load time

    @property
    def loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._loaded

    @property
    def model_name(self) -> str:
        """Model identifier string."""
        return f"moondream2 ({self._model_id})"

    @property
    def device(self) -> str:
        """Device the model is loaded on."""
        return self._device

    async def load(self, device: str | None = None) -> None:
        """Load the model and tokenizer into memory.

        Args:
            device: Force a specific device ("cuda", "cpu"). Auto-detected if None.

        Raises:
            VisionError: If loading fails.
        """
        if self._loaded:
            log.info("moondream_already_loaded")
            return

        start = time.monotonic()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine device
            if device is not None:
                self._device = device
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
                log.warning("moondream_cpu_fallback", reason="CUDA not available")

            log.info("moondream_loading", model=self._model_id, device=self._device)

            # Load in a thread to avoid blocking the event loop
            def _load() -> tuple[Any, Any]:
                model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                    device_map={"": self._device},
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id,
                    trust_remote_code=True,
                )
                return model, tokenizer

            self._model, self._tokenizer = await asyncio.get_event_loop().run_in_executor(
                None, _load
            )
            self._loaded = True

            elapsed = (time.monotonic() - start) * 1000
            log.info("moondream_loaded", device=self._device, elapsed_ms=round(elapsed, 1))

        except ImportError as exc:
            raise VisionError(
                f"Required package not installed: {exc}. Install torch and transformers.",
                operation="load",
                details={"model": self._model_id},
            ) from exc
        except Exception as exc:
            raise VisionError(
                f"Failed to load Moondream model: {exc}",
                operation="load",
                details={"model": self._model_id, "device": self._device},
            ) from exc

    async def unload(self) -> None:
        """Unload the model from memory, freeing VRAM/RAM."""
        if not self._loaded:
            return

        try:
            import torch

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._loaded = False
            log.info("moondream_unloaded")
        except Exception as exc:
            log.warning("moondream_unload_error", error=str(exc))
            self._loaded = False

    def _ensure_loaded(self) -> None:
        """Raise if the model is not loaded."""
        if not self._loaded or self._model is None:
            raise VisionError(
                "Moondream model is not loaded. Call load() first.",
                operation="check",
            )

    async def caption(self, image: Image.Image) -> VisionResult:
        """Generate a caption for an image.

        Args:
            image: PIL Image to caption.

        Returns:
            VisionResult with the caption text.
        """
        self._ensure_loaded()
        start = time.monotonic()

        def _infer() -> str:
            enc_image = self._model.encode_image(image)
            return self._model.answer_question(enc_image, "Describe this image.", self._tokenizer)  # type: ignore[no-any-return]

        try:
            text = await asyncio.get_event_loop().run_in_executor(None, _infer)
            elapsed = (time.monotonic() - start) * 1000

            return VisionResult(
                task=VisionTask.CAPTION,
                text=str(text).strip(),
                model_name=self.model_name,
                inference_time_ms=round(elapsed, 2),
                image_size=(image.width, image.height),
            )
        except Exception as exc:
            raise VisionError(
                f"Caption failed: {exc}",
                operation="caption",
                details={"image_size": f"{image.width}x{image.height}"},
            ) from exc

    async def ask(self, image: Image.Image, question: str) -> VisionResult:
        """Answer a question about an image (Visual Question Answering).

        Args:
            image: PIL Image to analyze.
            question: Natural language question about the image.

        Returns:
            VisionResult with the answer text.
        """
        self._ensure_loaded()
        start = time.monotonic()

        def _infer() -> str:
            enc_image = self._model.encode_image(image)
            return self._model.answer_question(enc_image, question, self._tokenizer)  # type: ignore[no-any-return]

        try:
            text = await asyncio.get_event_loop().run_in_executor(None, _infer)
            elapsed = (time.monotonic() - start) * 1000

            return VisionResult(
                task=VisionTask.VQA,
                text=str(text).strip(),
                model_name=self.model_name,
                inference_time_ms=round(elapsed, 2),
                image_size=(image.width, image.height),
                metadata={"question": question},
            )
        except Exception as exc:
            raise VisionError(
                f"VQA failed: {exc}",
                operation="vqa",
                details={"question": question, "image_size": f"{image.width}x{image.height}"},
            ) from exc

    async def detect(self, image: Image.Image, target: str) -> VisionResult:
        """Detect objects in an image matching a description.

        Moondream 2 supports basic object detection by asking positional questions.
        This method asks the model to locate objects and returns descriptions.

        Args:
            image: PIL Image to search.
            target: Object description to look for (e.g., "person", "cat", "red car").

        Returns:
            VisionResult with detection summary.
        """
        self._ensure_loaded()
        start = time.monotonic()

        detect_prompt = (
            f"Locate all instances of '{target}' in this image. "
            f"For each one, describe its position (left/right/center, top/bottom) "
            f"and approximate size relative to the image."
        )

        def _infer() -> str:
            enc_image = self._model.encode_image(image)
            return self._model.answer_question(enc_image, detect_prompt, self._tokenizer)  # type: ignore[no-any-return]

        try:
            text = await asyncio.get_event_loop().run_in_executor(None, _infer)
            elapsed = (time.monotonic() - start) * 1000

            return VisionResult(
                task=VisionTask.DETECT,
                text=str(text).strip(),
                model_name=self.model_name,
                inference_time_ms=round(elapsed, 2),
                image_size=(image.width, image.height),
                metadata={"target": target},
            )
        except Exception as exc:
            raise VisionError(
                f"Detection failed: {exc}",
                operation="detect",
                details={"target": target, "image_size": f"{image.width}x{image.height}"},
            ) from exc

    async def caption_from_path(self, image_path: Path) -> VisionResult:
        """Caption an image from a file path.

        Args:
            image_path: Path to the image file.

        Returns:
            VisionResult with the caption.
        """
        from PIL import Image as PILImage

        if not image_path.exists():
            raise VisionError(
                f"Image file not found: {image_path}",
                operation="caption",
                details={"path": str(image_path)},
            )

        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as exc:
            raise VisionError(
                f"Failed to open image: {exc}",
                operation="caption",
                details={"path": str(image_path)},
            ) from exc

        return await self.caption(image)

    def get_stats(self) -> dict[str, Any]:
        """Return model status and info.

        Returns:
            Dict with loaded state, model name, device, etc.
        """
        return {
            "loaded": self._loaded,
            "model_id": self._model_id,
            "model_name": self.model_name,
            "device": self._device,
        }
