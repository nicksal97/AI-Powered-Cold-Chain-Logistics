"""Agent tools.

Each tool is a thin, typed wrapper over the repository / domain layer. Inputs are
validated by the tool schema (enums, bounds), outputs are compact JSON, and failures
come back as a structured error the model can reason about instead of a stack trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel
from sqlalchemy import Engine

from coldchain.db import repository as repo
from coldchain.domain.weather import WeatherClient
from coldchain.logging import get_logger
from coldchain.rag.retriever import HybridRetriever

log = get_logger(__name__)

RuleId = Literal[
    "TEMP_BREACH", "FREEZE_EXCURSION", "PORT_CONGESTION", "TIER2_ESCALATION", "TEMP_ANOMALY"
]
SeverityLit = Literal["critical", "high", "medium"]


@dataclass
class ToolDeps:
    engine: Engine
    retriever: HybridRetriever
    weather: WeatherClient


def _dump(obj) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(exclude_none=True)
    if isinstance(obj, list):
        return json.dumps([o.model_dump(mode="json", exclude_none=True) for o in obj])
    return json.dumps(obj, default=str)


def _error(tool_name: str, exc: Exception) -> str:
    log.warning("tool.error", tool=tool_name, error=str(exc))
    return json.dumps({"error": type(exc).__name__, "detail": str(exc)[:300]})


def build_tools(deps: ToolDeps) -> list[BaseTool]:
    @tool
    def get_fleet_summary() -> str:
        """Fleet-wide snapshot: shipments by cargo profile and current risk class, and
        open incident counts by rule and severity. Start here for broad questions."""
        try:
            return _dump(repo.fleet_summary(deps.engine))
        except Exception as e:
            return _error("get_fleet_summary", e)

    @tool
    def list_alerts(
        severity: SeverityLit | None = None,
        rule_id: RuleId | None = None,
        shipment_id: str | None = None,
        status: Literal["open", "resolved", "any"] = "open",
        limit: int = 10,
    ) -> str:
        """List excursion incidents, most severe then most recent first. An incident is a
        run of consecutive readings violating one rule; it is open while the shipment's
        latest reading still violates it. Filter by severity, rule_id (TEMP_BREACH,
        FREEZE_EXCURSION, PORT_CONGESTION, TIER2_ESCALATION, TEMP_ANOMALY), shipment_id
        (e.g. SHP-0042) or status. limit is capped at 50. Each incident carries its peak
        value, duration in readings and the SOP-mandated recommended_action."""
        try:
            return _dump(
                repo.list_alerts(
                    deps.engine,
                    severity=severity,
                    rule_id=rule_id,
                    shipment_id=shipment_id,
                    status=status,
                    limit=limit,
                )
            )
        except Exception as e:
            return _error("list_alerts", e)

    @tool
    def get_shipment_status(shipment_id: str) -> str:
        """One shipment (e.g. SHP-0042): cargo profile and allowed temperature range, latest
        reading with GPS, last-24-readings temperature trend, open and recently resolved
        incidents."""
        try:
            return _dump(repo.shipment_status(deps.engine, shipment_id))
        except repo.ShipmentNotFound:
            return json.dumps({"error": "ShipmentNotFound", "detail": f"No shipment {shipment_id}"})
        except Exception as e:
            return _error("get_shipment_status", e)

    @tool
    def get_corridor_conditions(latitude: float, longitude: float) -> str:
        """Live weather at a GPS position (Open-Meteo) with a corridor risk rating
        (normal / elevated / high) and the factors behind it."""
        try:
            return _dump(deps.weather.corridor(latitude, longitude))
        except Exception as e:
            return _error("get_corridor_conditions", e)

    @tool
    def search_sop(query: str) -> str:
        """Search the cold-chain Standard Operating Procedures. Returns the most relevant
        passages with citations. Use it to justify every recommended action."""
        try:
            hits = deps.retriever.search(query)
            if not hits:
                return json.dumps({"results": [], "note": "No matching SOP passage."})
            return json.dumps(
                {
                    "results": [
                        {"citation": h.citation, "content": h.content, "score": round(h.score, 4)}
                        for h in hits
                    ]
                }
            )
        except Exception as e:
            return _error("search_sop", e)

    return [
        get_fleet_summary,
        list_alerts,
        get_shipment_status,
        get_corridor_conditions,
        search_sop,
    ]
