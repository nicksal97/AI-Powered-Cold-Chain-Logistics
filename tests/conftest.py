from __future__ import annotations

import os

# Must run before any coldchain import: models read the embedding dim at import time.
os.environ.update(
    {
        "COLDCHAIN_ENV": "test",
        "COLDCHAIN_EMBEDDING_PROVIDER": "hash",
        "COLDCHAIN_EMBEDDING_DIM": "256",
        "COLDCHAIN_CHECKPOINTER": "memory",
        "COLDCHAIN_LOG_JSON": "false",
        "COLDCHAIN_API_KEYS": "[]",
        "COLDCHAIN_DATABASE_URL": os.environ.get(
            "TEST_DATABASE_URL",
            "postgresql+psycopg://coldchain:coldchain@localhost:5432/coldchain_test",
        ),
    }
)

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from coldchain.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
SOP_DIR = ROOT / "data" / "policy"


class ScriptedChatModel(BaseChatModel):
    """Deterministic stand-in for an LLM: returns pre-scripted messages in order and
    records every prompt it receives, so tests can assert on the agent loop itself."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if not self.script:
            raise RuntimeError("ScriptedChatModel ran out of responses")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# --- synthetic telemetry with known violations ------------------------------------
def make_telemetry_csv(path: Path) -> Path:
    """3 trips x 128 hourly readings.
    SHP-0000 frozen, ends in an open TEMP_BREACH (last 3 readings at -1°C).
    SHP-0001 frozen, one resolved 2-reading breach in the middle, ends clean.
    SHP-0002 fresh (2°C), ends with port congestion 8.5 and a High Risk / 0.9 delay.
    """
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for trip in range(3):
        for i in range(128):
            temp, cong, risk, delay = -8.0 + (i % 3) * 0.5, 3.0, "Low Risk", 0.1
            if trip == 0 and i >= 125:
                temp = -1.0
            if trip == 1 and i in (60, 61):
                temp = 2.0
            if trip == 2:
                temp = 2.0 + (i % 3) * 0.3
                if i == 127:
                    cong, risk, delay = 8.5, "High Risk", 0.9
            rows.append(
                {
                    "timestamp": (t0 + timedelta(hours=trip * 128 + i)).isoformat(),
                    "vehicle_gps_latitude": 33.75,
                    "vehicle_gps_longitude": -118.2,
                    "iot_temperature": temp,
                    "cargo_condition_status": 0.5,
                    "risk_classification": risk,
                    "delay_probability": delay,
                    "port_congestion_level": cong,
                    "route_risk_level": 4.0,
                    "eta_variation_hours": 0.5,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _db_available(url: str) -> bool:
    try:
        admin = make_url(url).set(database="postgres")
        with create_engine(admin, isolation_level="AUTOCOMMIT").connect() as c:
            name = make_url(url).database
            if not c.execute(
                text("SELECT 1 FROM pg_database WHERE datname=:n"), {"n": name}
            ).scalar():
                c.execute(text(f'CREATE DATABASE "{name}"'))
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def seeded_engine(tmp_path_factory) -> Iterator:
    settings = get_settings()
    if not _db_available(settings.database_url):
        pytest.skip("Postgres not reachable; set TEST_DATABASE_URL")
    from coldchain.cli import migrate
    from coldchain.db.session import get_engine
    from coldchain.domain.detection import run_detection
    from coldchain.ingestion.sop import ingest_sop_dir
    from coldchain.ingestion.telemetry import ingest_telemetry
    from coldchain.rag.embeddings import build_embedder

    engine = get_engine(settings.database_url)
    with engine.begin() as c:
        c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    os.environ["COLDCHAIN_ROOT"] = str(ROOT)
    migrate()
    ingest_telemetry(make_telemetry_csv(tmp_path_factory.mktemp("data") / "t.csv"), engine)
    ingest_sop_dir(SOP_DIR, build_embedder(settings), engine)
    run_detection(engine)
    yield engine
    engine.dispose()
