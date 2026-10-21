from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from coldchain.api.container import Container

_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_container(request: Request) -> Container:
    return request.app.state.container


def require_api_key(
    key: str | None = Depends(_api_key), c: Container = Depends(get_container)
) -> None:
    allowed = c.settings.api_keys
    if not allowed:  # auth disabled (dev/test only; prod refuses to start without keys)
        return
    if not key or not any(secrets.compare_digest(key, k) for k in allowed):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
