"""Models route — model management, VRAM usage, model swapping.

Endpoints:
    GET  /api/models — List all models with status
    GET  /api/models/vram — VRAM usage breakdown
    POST /api/models/swap — Swap the active LLM model
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ara.api.app import AppState, get_app_state
from ara.api.schemas import ModelItem, ModelSwapRequest, ModelSwapResponse, VRAMUsageResponse
from ara.config.models import MODELS, ModelLocation, ModelRole
from ara.utils.logging import get_logger

log = get_logger("api.models")

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelItem])
async def list_models(
    state: AppState = Depends(get_app_state),
) -> list[ModelItem]:
    """List all registered models with their status."""
    items: list[ModelItem] = []

    for role, spec in MODELS.items():
        loaded = False

        # Check if model is actually loaded
        if role == ModelRole.PRIMARY_LLM and state.llm_client:
            loaded = state.llm_client.is_connected
        elif role == ModelRole.VISION_ALWAYS_ON and state.vision_pipeline:
            loaded = state.vision_pipeline.moondream_loaded

        items.append(
            ModelItem(
                role=str(role),
                name=spec.name,
                location=str(spec.location),
                vram_gb=spec.vram_gb,
                ram_mb=spec.ram_mb,
                loaded=loaded,
                description=spec.description,
            ),
        )

    return items


@router.get("/vram", response_model=VRAMUsageResponse)
async def vram_usage(
    state: AppState = Depends(get_app_state),
) -> VRAMUsageResponse:
    """Get current VRAM usage breakdown."""
    from ara.config.models import TOTAL_VRAM_GB

    loaded_models: list[ModelItem] = []
    total_used = 0.0

    for role, spec in MODELS.items():
        if spec.location != ModelLocation.GPU:
            continue

        loaded = False
        if role == ModelRole.PRIMARY_LLM and state.llm_client:
            loaded = state.llm_client.is_connected
        elif role == ModelRole.VISION_ALWAYS_ON and state.vision_pipeline:
            loaded = state.vision_pipeline.moondream_loaded

        if loaded:
            total_used += spec.vram_gb
            loaded_models.append(
                ModelItem(
                    role=str(role),
                    name=spec.name,
                    location="gpu",
                    vram_gb=spec.vram_gb,
                    loaded=True,
                    description=spec.description,
                ),
            )

    return VRAMUsageResponse(
        total_gb=TOTAL_VRAM_GB,
        used_gb=round(total_used, 1),
        free_gb=round(TOTAL_VRAM_GB - total_used, 1),
        models=loaded_models,
    )


@router.post("/swap", response_model=ModelSwapResponse)
async def swap_model(
    request: ModelSwapRequest,
    state: AppState = Depends(get_app_state),
) -> ModelSwapResponse:
    """Swap the active LLM to a different model.

    This triggers Ollama to load the new model (may take time).
    """
    if not state.llm_client:
        return ModelSwapResponse(
            previous_model="",
            current_model="",
            success=False,
            message="LLM client not initialized",
        )

    previous = state.llm_client.active_model
    start = time.monotonic()

    try:
        await state.llm_client.swap_model(request.target_model)
        elapsed = (time.monotonic() - start) * 1000

        log.info(
            "model_swapped",
            previous=previous,
            current=request.target_model,
            elapsed_ms=round(elapsed, 1),
        )

        return ModelSwapResponse(
            previous_model=previous,
            current_model=state.llm_client.active_model,
            success=True,
            elapsed_ms=round(elapsed, 1),
            message=f"Swapped from {previous} to {request.target_model}",
        )

    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        log.error("model_swap_failed", target=request.target_model, error=str(exc))
        return ModelSwapResponse(
            previous_model=previous,
            current_model=state.llm_client.active_model,
            success=False,
            elapsed_ms=round(elapsed, 1),
            message=str(exc),
        )
