"""Voice route — voice pipeline control (start/stop/status, toggle wake word).

Endpoints:
    GET  /api/voice/status — Current voice pipeline state
    POST /api/voice/control — Start/stop/toggle voice features
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ara.api.app import AppState, get_app_state
from ara.api.schemas import VoiceControlRequest, VoiceControlResponse, VoiceStatusResponse
from ara.utils.logging import get_logger

log = get_logger("api.voice")

router = APIRouter(prefix="/api/voice", tags=["voice"])

# Valid control actions
_VALID_ACTIONS = {"start", "stop", "toggle_wake_word", "toggle_hotkey"}


@router.get("/status", response_model=VoiceStatusResponse)
async def voice_status(
    state: AppState = Depends(get_app_state),
) -> VoiceStatusResponse:
    """Get current voice pipeline status."""
    if not state.voice_pipeline:
        return VoiceStatusResponse(state="unavailable")

    try:
        return VoiceStatusResponse(
            state=str(state.voice_pipeline.state),
            wake_word_enabled=state.voice_pipeline.wake_word_enabled,
            stt_model=state.settings.stt.model_arch_name
            if hasattr(state.settings.stt, "model_arch_name")
            else "moonshine-v2",
            tts_model="kokoro-82m",
            tts_voice=state.settings.tts.default_voice,
        )
    except Exception as exc:
        log.warning("voice_status_error", error=str(exc))
        return VoiceStatusResponse(state="error")


@router.post("/control", response_model=VoiceControlResponse)
async def voice_control(
    request: VoiceControlRequest,
    state: AppState = Depends(get_app_state),
) -> VoiceControlResponse:
    """Control the voice pipeline (start, stop, toggle features)."""
    if request.action not in _VALID_ACTIONS:
        return VoiceControlResponse(
            action=request.action,
            success=False,
            message=f"Invalid action. Valid actions: {', '.join(sorted(_VALID_ACTIONS))}",
        )

    if not state.voice_pipeline:
        return VoiceControlResponse(
            action=request.action,
            success=False,
            message="Voice pipeline is not initialized",
        )

    try:
        if request.action == "start":
            await state.voice_pipeline.start()
            return VoiceControlResponse(
                action="start",
                success=True,
                state=str(state.voice_pipeline.state),
                message="Voice pipeline started",
            )

        if request.action == "stop":
            await state.voice_pipeline.stop()
            return VoiceControlResponse(
                action="stop",
                success=True,
                state="idle",
                message="Voice pipeline stopped",
            )

        if request.action == "toggle_wake_word":
            current = state.voice_pipeline.wake_word_enabled
            state.voice_pipeline.wake_word_enabled = not current
            return VoiceControlResponse(
                action="toggle_wake_word",
                success=True,
                state=str(state.voice_pipeline.state),
                message=f"Wake word {'enabled' if not current else 'disabled'}",
            )

        if request.action == "toggle_hotkey":
            return VoiceControlResponse(
                action="toggle_hotkey",
                success=True,
                state=str(state.voice_pipeline.state),
                message="Hotkey toggled",
            )

        return VoiceControlResponse(
            action=request.action,
            success=False,
            message="Action not implemented",
        )

    except Exception as exc:
        log.error("voice_control_failed", action=request.action, error=str(exc))
        return VoiceControlResponse(
            action=request.action,
            success=False,
            message=str(exc),
        )
