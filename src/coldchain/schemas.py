"""Shared response models used by both the HTTP API and the agent tools, so the
agent sees exactly the same data contract a human client does."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ReadingOut(_Model):
    recorded_at: datetime
    latitude: float
    longitude: float
    temperature_c: float
    risk_class: str | None
    delay_probability: float | None
    port_congestion: float | None
    eta_variation_h: float | None


class AlertOut(_Model):
    """An excursion incident (one or more consecutive violating readings)."""

    id: int
    shipment_id: str
    rule_id: str
    severity: str
    status: str
    message: str
    recommended_action: str
    sop_reference: str
    threshold: float | None
    peak_value: float | None
    readings_count: int
    started_at: datetime
    ended_at: datetime


class TempStats(BaseModel):
    window_readings: int
    min_c: float
    max_c: float
    mean_c: float
    pct_out_of_range: float


class ShipmentStatus(BaseModel):
    shipment_id: str
    cargo_profile: str
    allowed_range_c: str
    latest: ReadingOut
    temperature_trend: TempStats
    open_incidents: list[AlertOut]
    recent_resolved: list[AlertOut]


class FleetSummary(BaseModel):
    as_of: datetime | None
    shipments: int
    readings: int
    shipments_by_profile: dict[str, int]
    shipments_by_risk: dict[str, int]
    open_incidents_by_rule: dict[str, int]
    open_incidents_by_severity: dict[str, int]


class SopHit(BaseModel):
    citation: str
    content: str
    score: float


class CorridorConditions(BaseModel):
    latitude: float
    longitude: float
    air_temperature_c: float | None
    wind_speed_kmh: float | None
    precipitation_mm: float | None
    weather_code: int | None
    conditions: str
    corridor_risk: str
    risk_factors: list[str]
