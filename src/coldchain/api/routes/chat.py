from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from coldchain.api.container import Container
from coldchain.api.deps import get_container, require_api_key
from coldchain.api.metrics import AGENT_LATENCY, AGENT_RUNS, TOOL_CALLS
from coldchain.api.schemas import ChatRequest, ChatResponse, ToolCallTrace
from coldchain.logging import get_logger

router = APIRouter(prefix="/v1/chat", tags=["chat"], dependencies=[Depends(require_api_key)])
log = get_logger(__name__)


def _require_agent(c: Container) -> None:
    if c.agent is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Assistant disabled: no LLM configured (set COLDCHAIN_LLM_API_KEY).",
        )


def _config(c: Container, thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": c.settings.agent_max_steps * 2 + 1,
    }


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, c: Container = Depends(get_container)) -> ChatResponse:
    _require_agent(c)
    thread_id = req.thread_id or uuid.uuid4().hex
    t0 = time.perf_counter()
    try:
        result = c.agent.invoke({"messages": [HumanMessage(req.message)]}, _config(c, thread_id))
    except GraphRecursionError as e:
        AGENT_RUNS.labels("step_limit").inc()
        raise HTTPException(
            422,
            "The agent hit its step budget without an answer. Try a narrower question.",
        ) from e
    except Exception as e:
        AGENT_RUNS.labels("error").inc()
        log.exception("agent.failed", thread_id=thread_id)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Agent backend failed.") from e

    # only report this turn's tool calls, not the whole thread history
    messages = result["messages"]
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn = messages[last_human + 1 :]
    calls = [
        ToolCallTrace(name=tc["name"], args=tc["args"])
        for m in turn
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]
    for tc in calls:
        TOOL_CALLS.labels(tc.name).inc()
    elapsed = time.perf_counter() - t0
    AGENT_LATENCY.observe(elapsed)
    AGENT_RUNS.labels("ok").inc()
    log.info(
        "agent.ok", thread_id=thread_id, tools=[t.name for t in calls], seconds=round(elapsed, 2)
    )
    return ChatResponse(
        thread_id=thread_id,
        answer=str(turn[-1].content) if turn else "",
        tool_calls=calls,
        latency_ms=int(elapsed * 1000),
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/stream")
def chat_stream(req: ChatRequest, c: Container = Depends(get_container)) -> StreamingResponse:
    """Server-sent events: `thread`, `tool_call`, `tool_result`, `answer`, `error`, `done`."""
    _require_agent(c)
    thread_id = req.thread_id or uuid.uuid4().hex

    def gen() -> Iterator[str]:
        yield _sse("thread", {"thread_id": thread_id})
        try:
            for update in c.agent.stream(
                {"messages": [HumanMessage(req.message)]},
                _config(c, thread_id),
                stream_mode="updates",
            ):
                for node, state in update.items():
                    for m in state.get("messages", []):
                        if node == "reasoner" and isinstance(m, AIMessage):
                            for tc in m.tool_calls:
                                TOOL_CALLS.labels(tc["name"]).inc()
                                yield _sse("tool_call", {"name": tc["name"], "args": tc["args"]})
                            if not m.tool_calls:
                                yield _sse("answer", {"content": m.content})
                        elif isinstance(m, ToolMessage):
                            yield _sse(
                                "tool_result", {"name": m.name, "preview": str(m.content)[:400]}
                            )
            AGENT_RUNS.labels("ok").inc()
        except GraphRecursionError:
            AGENT_RUNS.labels("step_limit").inc()
            yield _sse("error", {"detail": "step budget exceeded"})
        except Exception:
            AGENT_RUNS.labels("error").inc()
            log.exception("agent.stream_failed", thread_id=thread_id)
            yield _sse("error", {"detail": "agent backend failed"})
        yield _sse("done", {})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
