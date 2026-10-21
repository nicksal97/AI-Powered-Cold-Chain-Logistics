"""Read-side queries. Every query the agent can trigger lives here as a typed,
parameterised function with a bounded result size. There is no path from LLM
output to raw SQL."""

from __future__ import annotations

from sqlalchemy import Engine, case, func, select

from coldchain.db.models import Alert, Shipment, TelemetryReading
from coldchain.db.session import session_scope
from coldchain.domain.rules import PROFILES, SEVERITY_ORDER, Severity
from coldchain.schemas import (
    AlertOut,
    FleetSummary,
    ReadingOut,
    ShipmentStatus,
    TempStats,
)

MAX_LIMIT = 50
_severity_rank = case(
    *((Alert.severity == str(s), rank) for s, rank in SEVERITY_ORDER.items()), else_=99
)


class ShipmentNotFound(LookupError):
    pass


def list_alerts(
    engine: Engine | None = None,
    *,
    severity: str | None = None,
    rule_id: str | None = None,
    shipment_id: str | None = None,
    status: str = "open",
    limit: int = 10,
) -> list[AlertOut]:
    """Incidents, most severe first, then most recent."""
    stmt = select(Alert)
    if status != "any":
        stmt = stmt.where(Alert.status == status)
    if severity:
        stmt = stmt.where(Alert.severity == severity.lower())
    if rule_id:
        stmt = stmt.where(Alert.rule_id == rule_id.upper())
    if shipment_id:
        stmt = stmt.where(Alert.shipment_id == shipment_id.strip().upper())
    stmt = stmt.order_by(_severity_rank, Alert.ended_at.desc(), Alert.id.desc())
    stmt = stmt.limit(max(1, min(limit, MAX_LIMIT)))
    with session_scope(engine) as s:
        return [AlertOut.model_validate(a) for a in s.execute(stmt).scalars()]


def shipment_status(engine: Engine | None, shipment_id: str, window: int = 24) -> ShipmentStatus:
    sid = shipment_id.strip().upper()
    with session_scope(engine) as s:
        ship = s.get(Shipment, sid)
        if ship is None:
            raise ShipmentNotFound(sid)
        recent = list(
            s.execute(
                select(TelemetryReading)
                .where(TelemetryReading.shipment_id == sid)
                .order_by(TelemetryReading.recorded_at.desc())
                .limit(window)
            ).scalars()
        )
    band = PROFILES[ship.cargo_profile]
    temps = [r.temperature_c for r in recent]
    bad = sum(1 for t in temps if t > band.max_c or (band.min_c is not None and t < band.min_c))
    lo = "" if band.min_c is None else f"{band.min_c:.1f}"
    return ShipmentStatus(
        shipment_id=sid,
        cargo_profile=ship.cargo_profile,
        allowed_range_c=f"{lo or '≤'} {'to ' if lo else ''}{band.max_c:.1f}".strip(),
        latest=ReadingOut.model_validate(recent[0]),
        temperature_trend=TempStats(
            window_readings=len(temps),
            min_c=round(min(temps), 2),
            max_c=round(max(temps), 2),
            mean_c=round(sum(temps) / len(temps), 2),
            pct_out_of_range=round(100 * bad / len(temps), 1),
        ),
        open_incidents=list_alerts(engine, shipment_id=sid, status="open", limit=10),
        recent_resolved=list_alerts(engine, shipment_id=sid, status="resolved", limit=5),
    )


def fleet_summary(engine: Engine | None = None) -> FleetSummary:
    with session_scope(engine) as s:
        as_of, readings = s.execute(
            select(func.max(TelemetryReading.recorded_at), func.count())
        ).one()
        shipments = s.scalar(select(func.count()).select_from(Shipment)) or 0
        by_profile = dict(
            s.execute(
                select(Shipment.cargo_profile, func.count()).group_by(Shipment.cargo_profile)
            ).all()
        )
        ranked = select(
            TelemetryReading.risk_class,
            func.row_number()
            .over(
                partition_by=TelemetryReading.shipment_id,
                order_by=TelemetryReading.recorded_at.desc(),
            )
            .label("rn"),
        ).subquery()
        by_risk = dict(
            s.execute(
                select(ranked.c.risk_class, func.count())
                .where(ranked.c.rn == 1)
                .group_by(ranked.c.risk_class)
            ).all()
        )
        open_ = Alert.status.in_(["open", "acknowledged"])
        by_rule = dict(
            s.execute(
                select(Alert.rule_id, func.count()).where(open_).group_by(Alert.rule_id)
            ).all()
        )
        by_sev = dict(
            s.execute(
                select(Alert.severity, func.count()).where(open_).group_by(Alert.severity)
            ).all()
        )
    order = [str(x) for x in Severity]
    return FleetSummary(
        as_of=as_of,
        shipments=shipments,
        readings=readings,
        shipments_by_profile=by_profile,
        shipments_by_risk={str(k): v for k, v in by_risk.items()},
        open_incidents_by_rule=by_rule,
        open_incidents_by_severity={k: by_sev[k] for k in order if k in by_sev},
    )


def acknowledge(engine: Engine | None, alert_id: int) -> AlertOut | None:
    """Mark an open incident as acknowledged. Resolved incidents are left alone."""
    with session_scope(engine) as s:
        a = s.get(Alert, alert_id)
        if a is None:
            return None
        if a.status == "open":
            a.status = "acknowledged"
            s.flush()
            s.refresh(a)
        return AlertOut.model_validate(a)
