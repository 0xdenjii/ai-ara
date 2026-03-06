"""FastAPI backend — REST + WebSocket API, routes, schemas.

Exports:
    create_app: FastAPI application factory.
    AppState: Shared application state (injected into all routes).
    get_app_state: Dependency injection helper for routes.
"""

from ara.api.app import AppState, create_app, get_app_state

__all__ = [
    "AppState",
    "create_app",
    "get_app_state",
]
