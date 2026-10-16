"""Vision route — image analysis, OCR, document analysis.

Endpoints:
    POST /api/vision/analyze — Analyze an uploaded image
    GET  /api/vision/status — Vision pipeline status
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from ara.api.app import AppState, get_app_state
from ara.utils.logging import get_logger
from ara.vision.pipeline import VisionTaskType

log = get_logger("api.vision")

router = APIRouter(prefix="/api/vision", tags=["vision"])

# Maximum file size: 20 MB
_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024

# Allowed MIME types
_ALLOWED_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp",
    }
)


# ── Response models ──────────────────────────────────────────────────────────


class VisionAnalyzeResponse(BaseModel):
    """Response from vision analysis."""

    description: str = ""
    task: str = ""
    model_used: str = ""
    elapsed_ms: float = 0.0
    success: bool = False
    error: str = ""


class VisionStatusResponse(BaseModel):
    """Vision pipeline status."""

    available: bool = False
    models_loaded: list[str] = Field(default_factory=list)
    always_on: bool = False


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/status", response_model=VisionStatusResponse)
async def vision_status(
    state: AppState = Depends(get_app_state),
) -> VisionStatusResponse:
    """Get vision pipeline status."""
    if not state.vision_pipeline:
        return VisionStatusResponse(available=False)

    try:
        loaded: list[str] = []
        if state.vision_pipeline._moondream and state.vision_pipeline._moondream._loaded:
            loaded.append("moondream")
        if state.vision_pipeline._qwen_vl and state.vision_pipeline._qwen_vl._loaded:
            loaded.append("qwen-vl")

        return VisionStatusResponse(
            available=True,
            models_loaded=loaded,
            always_on=state.settings.vision.vision_always_on,
        )
    except Exception as exc:
        log.warning("vision_status_error", error=str(exc))
        return VisionStatusResponse(available=False)


@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def vision_analyze(
    file: UploadFile = File(...),
    question: str = Form(default="Describe this image in detail."),
    task: str = Form(default="caption"),
    state: AppState = Depends(get_app_state),
) -> VisionAnalyzeResponse:
    """Analyze an uploaded image.

    Args:
        file: Image file (JPEG, PNG, WebP, GIF, BMP)
        question: Question to ask about the image (for VQA task)
        task: Analysis task — "caption", "vqa", "ocr", "detect", "document"
    """
    if not state.vision_pipeline:
        return VisionAnalyzeResponse(
            error="Vision pipeline is not initialized",
            task=task,
        )

    # Validate MIME type
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_MIME_TYPES:
        return VisionAnalyzeResponse(
            error=f"Unsupported file type: {content_type}. Allowed: {', '.join(sorted(_ALLOWED_MIME_TYPES))}",
            task=task,
        )

    # Read file content with chunked size check to prevent memory exhaustion
    try:
        chunks: list[bytes] = []
        total_read = 0
        while True:
            chunk = await file.read(65536)  # 64KB chunks
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > _MAX_FILE_SIZE_BYTES:
                return VisionAnalyzeResponse(
                    error=f"File too large (>{_MAX_FILE_SIZE_BYTES} bytes). Maximum: {_MAX_FILE_SIZE_BYTES} bytes",
                    task=task,
                )
            chunks.append(chunk)
        content = b"".join(chunks)
    except Exception as exc:
        return VisionAnalyzeResponse(
            error=f"Failed to read uploaded file: {exc}",
            task=task,
        )

    if not content:
        return VisionAnalyzeResponse(
            error="Uploaded file is empty",
            task=task,
        )

    # Save to temp file for the vision pipeline
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=str(state.settings.data_dir)
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        import time

        # Open image as PIL Image for the pipeline
        pil_image = Image.open(tmp_path)
        pil_image.load()  # Force load so file handle can be released

        start = time.monotonic()

        # Map task string to VisionTaskType enum
        task_type_map: dict[str, VisionTaskType] = {
            "caption": VisionTaskType.CAPTION,
            "vqa": VisionTaskType.VQA,
            "ocr": VisionTaskType.OCR,
            "document": VisionTaskType.DOCUMENT,
            "detect": VisionTaskType.DETECT,
        }
        vision_task = task_type_map.get(task)
        if vision_task is None:
            return VisionAnalyzeResponse(
                error=f"Unknown task: {task}. Supported: caption, vqa, ocr, document, detect",
                task=task,
            )

        # Route to appropriate pipeline method based on task
        result: Any = None
        model_used = ""

        if task == "caption":
            result = await state.vision_pipeline.analyze(
                pil_image, vision_task, question="Describe this image."
            )
        elif task == "vqa":
            result = await state.vision_pipeline.analyze(pil_image, vision_task, question=question)
        elif task == "ocr":
            result = await state.vision_pipeline.analyze(
                pil_image, vision_task, question="Extract all text from this image."
            )
        elif task == "document":
            result = await state.vision_pipeline.analyze(
                pil_image,
                vision_task,
                question="Analyze this document. Describe its contents, structure, and key information.",
            )
        elif task == "detect":
            result = await state.vision_pipeline.analyze(
                pil_image,
                vision_task,
                question="Identify and list all objects in this image with their locations.",
            )

        elapsed = (time.monotonic() - start) * 1000

        # Extract response text from result (PipelineResult has .text, .model_used, .total_time_ms)
        description = ""
        if result:
            description = result.text
            model_used = result.model_used

        return VisionAnalyzeResponse(
            description=description,
            task=task,
            model_used=model_used,
            elapsed_ms=round(result.total_time_ms if result else elapsed, 1),
            success=True,
        )

    except Exception as exc:
        log.error("vision_analyze_failed", error=str(exc), task=task)
        return VisionAnalyzeResponse(
            error=f"Vision analysis failed: {exc}",
            task=task,
        )
    finally:
        # Clean up temp file
        if tmp_path and tmp_path.exists():
            import contextlib

            with contextlib.suppress(OSError):
                tmp_path.unlink()
