"""Wires the object graph once at startup. Tests build their own Container with
fakes, so nothing in the app reaches for globals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import Engine

from coldchain.agent.graph import build_agent
from coldchain.agent.llm import build_llm
from coldchain.agent.tools import ToolDeps, build_tools
from coldchain.config import Settings
from coldchain.db.session import get_engine
from coldchain.domain.weather import WeatherClient
from coldchain.logging import get_logger
from coldchain.rag.embeddings import build_embedder
from coldchain.rag.retriever import HybridRetriever

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    engine: Engine
    retriever: HybridRetriever
    weather: WeatherClient
    agent: Any | None  # None when no LLM is configured; ops endpoints still work
    _closers: list = field(default_factory=list)

    def close(self) -> None:
        for c in reversed(self._closers):
            c()


def build_container(
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
    engine: Engine | None = None,
    weather: WeatherClient | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Container:
    engine = engine or get_engine(settings.database_url)
    retriever = HybridRetriever(
        engine, build_embedder(settings), settings.retrieval_k, settings.retrieval_candidates
    )
    weather = weather or WeatherClient(settings.weather_base_url, settings.weather_timeout_s)
    closers: list = [weather.close]

    if checkpointer is None:
        if settings.checkpointer == "postgres":
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                settings.psycopg_dsn,
                max_size=settings.db_pool_size,
                open=True,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            )
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            closers.append(pool.close)
        else:
            checkpointer = InMemorySaver()

    tools = build_tools(ToolDeps(engine=engine, retriever=retriever, weather=weather))
    if llm is None:
        try:
            llm = build_llm(settings)
        except ValueError as e:
            log.warning("agent.disabled", reason=str(e))
    agent = build_agent(llm, tools, checkpointer) if llm is not None else None
    return Container(settings, engine, retriever, weather, agent, closers)
