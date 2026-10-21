"""Incident detection job.

For each shipment: evaluate the SOP rules plus the statistical anomaly layer on every
reading, collapse consecutive violations into episodes, and upsert them. Idempotent
(keyed on shipment+rule+start), so it is safe to run after every telemetry load or on
a schedule. An acknowledged incident stays acknowledged until it resolves.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import groupby

from sqlalchemy import Engine, case, select
from sqlalchemy.dialects.postgresql import insert

from coldchain.db.models import Alert, Shipment, TelemetryReading
from coldchain.db.session import session_scope
from coldchain.domain.rules import (
    Episode,
    Reading,
    Severity,
    Violation,
    build_episodes,
    temperature_anomalies,
)
from coldchain.logging import get_logger

log = get_logger(__name__)
BATCH = 2000


def anomaly_violations(readings: list[Reading]) -> dict[int, list[Violation]]:
    out: dict[int, list[Violation]] = defaultdict(list)
    for r, z in temperature_anomalies(readings):
        out[r.reading_id].append(
            Violation(
                rule_id="TEMP_ANOMALY",
                severity=Severity.MEDIUM,
                message=(
                    f"Temperature {r.temperature_c:.1f}°C is a statistical outlier for this "
                    f"shipment (robust z={z})."
                ),
                recommended_action="Verify sensor calibration and door-open events before acting.",
                sop_reference="Statistical monitoring (advisory, not SOP-mandated)",
                observed_value=r.temperature_c,
                threshold=None,
            )
        )
    return out


def _row(ep: Episode) -> dict:
    return {
        "shipment_id": ep.shipment_id,
        "rule_id": ep.rule_id,
        "severity": str(ep.severity),
        "message": ep.summary(),
        "recommended_action": ep.peak.recommended_action,
        "sop_reference": ep.sop_reference,
        "threshold": ep.threshold,
        "peak_value": ep.peak_value,
        "readings_count": ep.count,
        "started_at": ep.first.recorded_at,
        "ended_at": ep.last.recorded_at,
        "first_reading_id": ep.first.reading_id,
        "last_reading_id": ep.last.reading_id,
        "status": "open" if ep.is_open else "resolved",
    }


def _upsert(session, rows: list[dict]) -> None:
    if not rows:
        return
    stmt = insert(Alert).values(rows)
    ex = stmt.excluded
    stmt = stmt.on_conflict_do_update(
        constraint="uq_alert_episode",
        set_={
            "message": ex.message,
            "recommended_action": ex.recommended_action,
            "peak_value": ex.peak_value,
            "readings_count": ex.readings_count,
            "ended_at": ex.ended_at,
            "last_reading_id": ex.last_reading_id,
            "status": case(
                (ex.status == "resolved", "resolved"),
                (Alert.status == "acknowledged", "acknowledged"),
                else_=ex.status,
            ),
        },
    )
    session.execute(stmt)


def run_detection(engine: Engine | None = None) -> dict[str, int]:
    stats: Counter[str] = Counter()
    pending: list[dict] = []
    with session_scope(engine) as session:
        q = (
            select(TelemetryReading, Shipment.cargo_profile)
            .join(Shipment, Shipment.shipment_id == TelemetryReading.shipment_id)
            .order_by(TelemetryReading.shipment_id, TelemetryReading.recorded_at)
            .execution_options(yield_per=BATCH)
        )
        for sid, group in groupby(session.execute(q), key=lambda row: row[0].shipment_id):
            readings = [
                Reading(
                    reading_id=t.id,
                    shipment_id=sid,
                    recorded_at=t.recorded_at,
                    temperature_c=t.temperature_c,
                    cargo_profile=profile,
                    risk_class=t.risk_class,
                    delay_probability=t.delay_probability,
                    port_congestion=t.port_congestion,
                    eta_variation_h=t.eta_variation_h,
                )
                for t, profile in group
            ]
            for ep in build_episodes(readings, extra=anomaly_violations(readings)):
                stats[f"{ep.rule_id}.{'open' if ep.is_open else 'resolved'}"] += 1
                pending.append(_row(ep))
            if len(pending) >= BATCH:
                _upsert(session, pending)
                pending.clear()
        _upsert(session, pending)
    log.info("detection.complete", **stats)
    return dict(sorted(stats.items()))
