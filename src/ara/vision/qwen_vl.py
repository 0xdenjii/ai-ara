"""Qwen-VL wrapper — on-demand heavy vision model via Ollama.

Provides:
    - QwenVLModel: Complex vision tasks (OCR, document analysis, detailed descriptions)
    - Uses Ollama's multimodal API (base64 image + text prompt)
    - Hot-swap lifecycle: load/unload/swap with primary LLM (~5GB VRAM)

Architecture:
    Qwen2.5-VL-7B is a powerful vision-language model that excels at OCR,
    document understanding, and complex visual reasoning.  It requires ~5GB
    VRAM, which means the primary LLM (16.5GB) must be unloaded first.

    We use Ollama's vision API for inference:
        POST /api/generate with {"model": "...", "prompt": "...", "images": ["base64..."]}

    This avoids loading a separate transformers model and leverages Ollama's
    existing GPU management for hot-swapping.

Usage:
    model = QwenVLModel(settings)
    await model.load()  # Pulls model into Ollama if needed
    result = await model.analyze(image, "What text is visible in this document?")
    await model.unload()
"""

from __future__ import annotations

import base64
import io
import time
from typing import TYPE_CHECKING, Any

import httpx

from ara.utils.errors import VisionError
from ara.utils.logging import get_logger
from ara.vision.moondream import VisionResult, VisionTask

if TYPE_CHECKING:
    from PIL import Image

    from ara.config.settings import AraSettings

log = get_logger("qwen_vl")

# Ollama API base URL
_OLLAMA_BASE = "http://localhost:11434"

# Default Ollama model name for Qwen-VL
_DEFAULT_OLLAMA_MODEL = "qwen2.5-vl:7b"


class QwenVLModel:
    """Qwen2.5-VL vision-language model wrapper via Ollama.

    Supports complex vision tasks: OCR, document analysis, detailed descriptions,
    and visual reasoning.  Uses Ollama's multimodal generate API.

    This model is loaded on-demand and requires hot-swapping with the primary LLM
    since both compete for GPU VRAM.

    Args:
        settings: Application settings.
        ollama_model: Ollama model name override (defaults to "qwen2.5-vl:7b").
    """

    def __init__(
        self,
        settings: AraSettings,
        *,
        ollama_model: str | None = None,
    ) -> None:
        self._settings = settings
        self._ollama_model = ollama_model or _DEFAULT_OLLAMA_MODEL
        self._loaded: bool = False
        self._ollama_base = _OLLAMA_BASE

    @property
    def loaded(self) -> bool:
        """Whether the model is available in Ollama."""
        return self._loaded

    @property
    def model_name(self) -> str:
        """Model identifier string."""
        return f"qwen-vl ({self._ollama_model})"

    async def load(self) -> None:
        """Ensure the model is available in Ollama.

        Checks if the model exists, and if not, attempts to pull it.
        Note: This does NOT load it into VRAM — Ollama does that on first inference.

        Raises:
            VisionError: If the model cannot be loaded.
        """
        if self._loaded:
            return

        try:
            # Check if model exists in Ollama
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._ollama_base}/api/show",
                    json={"name": self._ollama_model},
                )
                if resp.status_code == 200:
                    self._loaded = True
                    log.info("qwen_vl_available", model=self._ollama_model)
                    return

            # Model not found — try to pull
            log.info("qwen_vl_pulling", model=self._ollama_model)
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{self._ollama_base}/api/pull",
                    json={"name": self._ollama_model, "stream": False},
                )
                if resp.status_code == 200:
                    self._loaded = True
                    log.info("qwen_vl_pulled", model=self._ollama_model)
                else:
                    raise VisionError(
                        f"Failed to pull model '{self._ollama_model}': HTTP {resp.status_code}",
                        operation="load",
                        details={"model": self._ollama_model, "status": resp.status_code},
                    )

        except httpx.ConnectError as exc:
            raise VisionError(
                f"Cannot connect to Ollama at {self._ollama_base}. Is it running?",
                operation="load",
                details={"url": self._ollama_base},
            ) from exc
        except VisionError:
            raise
        except Exception as exc:
            raise VisionError(
                f"Failed to load Qwen-VL model: {exc}",
                operation="load",
                details={"model": self._ollama_model},
            ) from exc

    async def unload(self) -> None:
        """Unload the model from Ollama's VRAM.

        Sends a generate request with keep_alive=0 to force Ollama to free the model.
        """
        if not self._loaded:
            return

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self._ollama_base}/api/generate",
                    json={
                        "model": self._ollama_model,
                        "keep_alive": 0,
                    },
                )
            self._loaded = False
            log.info("qwen_vl_unloaded", model=self._ollama_model)
        except Exception as exc:
            log.warning("qwen_vl_unload_error", error=str(exc))
            self._loaded = False

    def _ensure_loaded(self) -> None:
        """Raise if the model is not available."""
        if not self._loaded:
            raise VisionError(
                "Qwen-VL model is not loaded. Call load() first.",
                operation="check",
            )

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert a PIL Image to a base64 string.

        Args:
            image: PIL Image.

        Returns:
            Base64-encoded PNG string.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def analyze(
        self,
        image: Image.Image,
        prompt: str = "Describe this image in detail.",
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> VisionResult:
        """Analyze an image with a text prompt.

        Args:
            image: PIL Image to analyze.
            prompt: The question or instruction about the image.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            VisionResult with the analysis text.
        """
        self._ensure_loaded()
        start = time.monotonic()

        image_b64 = self._image_to_base64(image)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._ollama_base}/api/generate",
                    json={
                        "model": self._ollama_model,
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            text = str(data.get("response", "")).strip()
            elapsed = (time.monotonic() - start) * 1000

            return VisionResult(
                task=VisionTask.VQA,
                text=text,
                model_name=self.model_name,
                inference_time_ms=round(elapsed, 2),
                image_size=(image.width, image.height),
                metadata={
                    "prompt": prompt,
                    "total_duration_ns": data.get("total_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                },
            )

        except httpx.ConnectError as exc:
            raise VisionError(
                f"Cannot connect to Ollama at {self._ollama_base}",
                operation="analyze",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise VisionError(
                f"Ollama request failed: HTTP {exc.response.status_code}",
                operation="analyze",
                details={"status": exc.response.status_code},
            ) from exc
        except Exception as exc:
            raise VisionError(
                f"Qwen-VL analysis failed: {exc}",
                operation="analyze",
                details={"prompt": prompt[:100]},
            ) from exc

    async def ocr(self, image: Image.Image) -> VisionResult:
        """Extract text from an image using OCR capabilities.

        Args:
            image: PIL Image containing text.

        Returns:
            VisionResult with extracted text.
        """
        return await self.analyze(
            image,
            prompt=(
                "Extract all visible text from this image. "
                "Preserve the layout and formatting as much as possible. "
                "If there are multiple sections, separate them clearly."
            ),
            max_tokens=2048,
            temperature=0.1,
        )

    async def analyze_document(self, image: Image.Image) -> VisionResult:
        """Analyze a document image (PDF page, screenshot, etc.).

        Args:
            image: PIL Image of the document.

        Returns:
            VisionResult with document analysis.
        """
        return await self.analyze(
            image,
            prompt=(
                "Analyze this document image. Provide: "
                "1) Document type (form, letter, receipt, code, etc.) "
                "2) Key information and data points "
                "3) Any notable formatting or layout details "
                "4) A brief summary of the content"
            ),
            max_tokens=2048,
            temperature=0.2,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return model status and info."""
        return {
            "loaded": self._loaded,
            "model_name": self.model_name,
            "ollama_model": self._ollama_model,
            "ollama_base": self._ollama_base,
        }
