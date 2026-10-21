"""Plain REST endpoints over the same repository the agent uses. Dashboards and
other services should hit these directly; they are cheaper and deterministic."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from coldchain.api.container import Container
from coldchain.api.deps import get_container, require_api_key
from coldchain.db import repository as repo
from coldchain.schemas import AlertOut, FleetSummary, ShipmentStatus, SopHit

router = APIRouter(prefix="/v1", tags=["operations"], dependencies=[Depends(require_api_key)])


@router.get("/fleet/summary", response_model=FleetSummary)
def fleet_summary(c: Container = Depends(get_container)) -> FleetSummary:
    return repo.fleet_summary(c.engine)


@router.get("/incidents", response_model=list[AlertOut])
def alerts(
    severity: Literal["critical", "high", "medium"] | None = None,
    rule_id: str | None = Query(default=None, max_length=48),
    shipment_id: str | None = Query(default=None, max_length=32),
    status: Literal["open", "acknowledged", "resolved", "any"] = "open",
    limit: int = Query(default=20, ge=1, le=repo.MAX_LIMIT),
    c: Container = Depends(get_container),
) -> list[AlertOut]:
    return repo.list_alerts(
        c.engine,
        severity=severity,
        rule_id=rule_id,
        shipment_id=shipment_id,
        status=status,
        limit=limit,
    )


@router.get("/shipments/{shipment_id}", response_model=ShipmentStatus)
def shipment(shipment_id: str, c: Container = Depends(get_container)) -> ShipmentStatus:
    try:
        return repo.shipment_status(c.engine, shipment_id)
    except repo.ShipmentNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"shipment {shipment_id} not found") from e


@router.get("/sop/search", response_model=list[SopHit])
def sop_search(
    q: str = Query(min_length=2, max_length=500),
    k: int = Query(default=4, ge=1, le=10),
    c: Container = Depends(get_container),
) -> list[SopHit]:
    return [
        SopHit(citation=h.citation, content=h.content, score=h.score)
        for h in c.retriever.search(q, k)
    ]


@router.post("/incidents/{incident_id}/ack", response_model=AlertOut)
def acknowledge(incident_id: int, c: Container = Depends(get_container)) -> AlertOut:
    out = repo.acknowledge(c.engine, incident_id)
    if out is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"incident {incident_id} not found")
    return out
