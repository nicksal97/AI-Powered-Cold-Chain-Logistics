from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Reuse to continue a conversation; omitted = new thread.",
    )


class ToolCallTrace(BaseModel):
    name: str
    args: dict


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    tool_calls: list[ToolCallTrace]
    latency_ms: int


class Health(BaseModel):
    status: str
    version: str
    database: bool | None = None
