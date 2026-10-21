"""Relational schema.

Three tables:
  telemetry_readings - one row per IoT reading per shipment (the fact table)
  alerts             - SOP rule violations, materialised by the detection job
  sop_chunks         - chunked SOP text with a pgvector embedding and an FTS column

The agent never touches these with free-form SQL. It goes through the typed
repository functions in coldchain.db.repository.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from coldchain.config import get_settings

EMBEDDING_DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    pass


class Shipment(Base):
    __tablename__ = "shipments"

    shipment_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    cargo_profile: Mapped[str] = mapped_column(String(16), nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    readings: Mapped[int] = mapped_column(Integer, nullable=False)


class TelemetryReading(Base):
    __tablename__ = "telemetry_readings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    shipment_id: Mapped[str] = mapped_column(String(32), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    temperature_c: Mapped[float] = mapped_column(Float, nullable=False)
    cargo_condition_score: Mapped[float | None] = mapped_column(Float)
    risk_class: Mapped[str | None] = mapped_column(String(32))
    delay_probability: Mapped[float | None] = mapped_column(Float)
    port_congestion: Mapped[float | None] = mapped_column(Float)
    route_risk: Mapped[float | None] = mapped_column(Float)
    eta_variation_h: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("shipment_id", "recorded_at", name="uq_reading_shipment_ts"),
        Index("ix_reading_recorded_at", "recorded_at"),
    )


class Alert(Base):
    """One excursion episode: consecutive readings of a shipment violating one rule.

    Open while the shipment's latest reading still violates the rule, resolved as soon
    as an in-range reading arrives. This is what keeps the queue actionable: thousands
    of bad readings collapse into a handful of incidents with a duration and a peak.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    shipment_id: Mapped[str] = mapped_column(String(32), nullable=False)
    rule_id: Mapped[str] = mapped_column(String(48), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False)
    sop_reference: Mapped[str] = mapped_column(String(128), nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float)
    peak_value: Mapped[float | None] = mapped_column(Float)
    readings_count: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    first_reading_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_reading_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("shipment_id", "rule_id", "started_at", name="uq_alert_episode"),
        Index("ix_alert_status_severity", "status", "severity", "ended_at"),
        Index("ix_alert_shipment", "shipment_id"),
    )


class SopChunk(Base):
    __tablename__ = "sop_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document: Mapped[str] = mapped_column(String(256), nullable=False)
    section: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    tsv = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', section || ' ' || content)", persisted=True),
    )

    __table_args__ = (
        Index("ix_sop_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_sop_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
