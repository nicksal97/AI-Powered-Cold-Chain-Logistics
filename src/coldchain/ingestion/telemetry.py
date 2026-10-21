"""Load the fleet telemetry CSV into Postgres.

The source dataset has no shipment identifier, so consecutive hourly readings are
grouped into synthetic trips of READINGS_PER_SHIPMENT rows (about 5 days each).
That gives every shipment a real time series, which the per-shipment anomaly
detection and the "how is SHP-0042 trending?" questions both depend on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text
from sqlalchemy.dialects.postgresql import insert

from coldchain.db.models import TelemetryReading
from coldchain.db.session import session_scope
from coldchain.domain.rules import FROZEN_PROFILE_MEDIAN_C
from coldchain.logging import get_logger

log = get_logger(__name__)

READINGS_PER_SHIPMENT = 128
CHUNK = 5000

COLUMN_MAP = {
    "timestamp": "recorded_at",
    "vehicle_gps_latitude": "latitude",
    "vehicle_gps_longitude": "longitude",
    "iot_temperature": "temperature_c",
    "cargo_condition_status": "cargo_condition_score",
    "risk_classification": "risk_class",
    "delay_probability": "delay_probability",
    "port_congestion_level": "port_congestion",
    "route_risk_level": "route_risk",
    "eta_variation_hours": "eta_variation_h",
}
REQUIRED = ["recorded_at", "latitude", "longitude", "temperature_c"]

# Shipment dimension, rebuilt from the readings. Profile logic mirrors
# rules.infer_profile (kept in SQL so 30k+ rows never leave the database).
REFRESH_SHIPMENTS = text(
    """
    INSERT INTO shipments (shipment_id, cargo_profile, first_seen_at, last_seen_at, readings)
    SELECT shipment_id,
           CASE WHEN percentile_cont(0.5) WITHIN GROUP (ORDER BY temperature_c) < :frozen_below
                THEN 'frozen' ELSE 'fresh' END,
           min(recorded_at), max(recorded_at), count(*)
    FROM telemetry_readings
    GROUP BY shipment_id
    ON CONFLICT (shipment_id) DO UPDATE SET
        cargo_profile = EXCLUDED.cargo_profile,
        first_seen_at = EXCLUDED.first_seen_at,
        last_seen_at  = EXCLUDED.last_seen_at,
        readings      = EXCLUDED.readings
    """
)


def load_frame(csv_path: Path) -> pd.DataFrame:
    missing = set(COLUMN_MAP) - set(pd.read_csv(csv_path, nrows=0).columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

    df = pd.read_csv(csv_path, usecols=list(COLUMN_MAP)).rename(columns=COLUMN_MAP)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, errors="coerce")
    df = df.sort_values("recorded_at", kind="stable").reset_index(drop=True)
    df["shipment_id"] = [f"SHP-{i // READINGS_PER_SHIPMENT:04d}" for i in range(len(df))]

    before = len(df)
    df = df.dropna(subset=REQUIRED)
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]
    df = df[df["temperature_c"].between(-60, 60)]  # outside this is a sensor fault, not cargo
    dropped = before - len(df)
    if dropped:
        log.warning("telemetry.rows_rejected", count=dropped)
    return df


def ingest_telemetry(csv_path: Path, engine: Engine | None = None) -> int:
    df = load_frame(csv_path)
    records = df.astype(object).where(df.notna(), None).to_dict("records")
    with session_scope(engine) as session:
        for start in range(0, len(records), CHUNK):
            stmt = insert(TelemetryReading).values(records[start : start + CHUNK])
            session.execute(stmt.on_conflict_do_nothing(constraint="uq_reading_shipment_ts"))
        session.execute(REFRESH_SHIPMENTS, {"frozen_below": FROZEN_PROFILE_MEDIAN_C})
    log.info("telemetry.ingested", rows=len(records), shipments=df["shipment_id"].nunique())
    return len(records)
