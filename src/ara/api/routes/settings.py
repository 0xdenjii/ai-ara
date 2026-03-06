"""Settings route — CRUD for personality, desktop, internet config.

Endpoints:
    GET  /api/settings — Full settings snapshot
    GET  /api/settings/personality — Personality settings
    PUT  /api/settings/personality — Update personality settings
    GET  /api/settings/desktop — Desktop automation settings
    PUT  /api/settings/desktop — Update desktop settings
    GET  /api/settings/internet — Internet settings
    PUT  /api/settings/internet — Update internet settings
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ara.api.app import AppState, get_app_state
from ara.api.schemas import (
    DesktopSettingsResponse,
    DesktopUpdateRequest,
    InternetSettingsResponse,
    InternetUpdateRequest,
    PersonalitySettingsResponse,
    PersonalityUpdateRequest,
    SettingsResponse,
)
from ara.utils.logging import get_logger

log = get_logger("api.settings")

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings(
    state: AppState = Depends(get_app_state),
) -> SettingsResponse:
    """Get all modifiable settings."""
    s = state.settings
    return SettingsResponse(
        personality=PersonalitySettingsResponse(
            humor_dial=s.personality.humor_dial,
            voice_identity=s.personality.voice_identity,
            auto_dial_down_on_distress=s.personality.auto_dial_down_on_distress,
            distress_dial_reduction=s.personality.distress_dial_reduction,
        ),
        desktop=DesktopSettingsResponse(
            enabled=s.desktop.enabled,
            require_confirmation=s.desktop.require_confirmation,
            command_timeout_seconds=s.desktop.command_timeout_seconds,
        ),
        internet=InternetSettingsResponse(
            internet_enabled=s.internet.internet_enabled,
            searxng_url=s.internet.searxng_url,
            web_cache_ttl_s=s.internet.web_cache_ttl_s,
        ),
    )


# ── Personality ──────────────────────────────────────────────────────────────


@router.get("/personality", response_model=PersonalitySettingsResponse)
async def get_personality(
    state: AppState = Depends(get_app_state),
) -> PersonalitySettingsResponse:
    """Get personality settings."""
    p = state.settings.personality
    return PersonalitySettingsResponse(
        humor_dial=p.humor_dial,
        voice_identity=p.voice_identity,
        auto_dial_down_on_distress=p.auto_dial_down_on_distress,
        distress_dial_reduction=p.distress_dial_reduction,
    )


@router.put("/personality", response_model=PersonalitySettingsResponse)
async def update_personality(
    request: PersonalityUpdateRequest,
    state: AppState = Depends(get_app_state),
) -> PersonalitySettingsResponse:
    """Update personality settings (partial update)."""
    p = state.settings.personality

    if request.humor_dial is not None:
        p.humor_dial = request.humor_dial
        log.info("personality_updated", humor_dial=request.humor_dial)

    if request.auto_dial_down_on_distress is not None:
        p.auto_dial_down_on_distress = request.auto_dial_down_on_distress

    if request.distress_dial_reduction is not None:
        p.distress_dial_reduction = request.distress_dial_reduction

    return PersonalitySettingsResponse(
        humor_dial=p.humor_dial,
        voice_identity=p.voice_identity,
        auto_dial_down_on_distress=p.auto_dial_down_on_distress,
        distress_dial_reduction=p.distress_dial_reduction,
    )


# ── Desktop ──────────────────────────────────────────────────────────────────


@router.get("/desktop", response_model=DesktopSettingsResponse)
async def get_desktop(
    state: AppState = Depends(get_app_state),
) -> DesktopSettingsResponse:
    """Get desktop automation settings."""
    d = state.settings.desktop
    return DesktopSettingsResponse(
        enabled=d.enabled,
        require_confirmation=d.require_confirmation,
        command_timeout_seconds=d.command_timeout_seconds,
    )


@router.put("/desktop", response_model=DesktopSettingsResponse)
async def update_desktop(
    request: DesktopUpdateRequest,
    state: AppState = Depends(get_app_state),
) -> DesktopSettingsResponse:
    """Update desktop automation settings (partial update)."""
    d = state.settings.desktop

    if request.enabled is not None:
        d.enabled = request.enabled
        log.info("desktop_updated", enabled=request.enabled)

    if request.require_confirmation is not None:
        d.require_confirmation = request.require_confirmation

    return DesktopSettingsResponse(
        enabled=d.enabled,
        require_confirmation=d.require_confirmation,
        command_timeout_seconds=d.command_timeout_seconds,
    )


# ── Internet ─────────────────────────────────────────────────────────────────


@router.get("/internet", response_model=InternetSettingsResponse)
async def get_internet(
    state: AppState = Depends(get_app_state),
) -> InternetSettingsResponse:
    """Get internet connectivity settings."""
    i = state.settings.internet
    return InternetSettingsResponse(
        internet_enabled=i.internet_enabled,
        searxng_url=i.searxng_url,
        web_cache_ttl_s=i.web_cache_ttl_s,
    )


@router.put("/internet", response_model=InternetSettingsResponse)
async def update_internet(
    request: InternetUpdateRequest,
    state: AppState = Depends(get_app_state),
) -> InternetSettingsResponse:
    """Update internet settings (partial update)."""
    i = state.settings.internet

    if request.internet_enabled is not None:
        i.internet_enabled = request.internet_enabled
        log.info("internet_updated", enabled=request.internet_enabled)

    return InternetSettingsResponse(
        internet_enabled=i.internet_enabled,
        searxng_url=i.searxng_url,
        web_cache_ttl_s=i.web_cache_ttl_s,
    )
