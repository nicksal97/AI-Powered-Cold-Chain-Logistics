import pytest
from sqlalchemy import func, select

from coldchain.config import get_settings
from coldchain.db import repository as repo
from coldchain.db.models import Alert, Shipment
from coldchain.db.session import session_scope
from coldchain.domain.detection import run_detection
from coldchain.ingestion.sop import ingest_sop_dir
from coldchain.rag.embeddings import build_embedder
from coldchain.rag.retriever import HybridRetriever
from tests.conftest import SOP_DIR

pytestmark = pytest.mark.integration


def test_profiles_inferred_per_trip(seeded_engine):
    with session_scope(seeded_engine) as s:
        got = dict(s.execute(select(Shipment.shipment_id, Shipment.cargo_profile)).all())
    assert got == {"SHP-0000": "frozen", "SHP-0001": "frozen", "SHP-0002": "fresh"}


def test_open_and_resolved_incidents(seeded_engine):
    [breach] = repo.list_alerts(seeded_engine, shipment_id="SHP-0000", rule_id="TEMP_BREACH")
    assert breach.status == "open" and breach.readings_count == 3 and breach.peak_value == -1.0

    resolved = {
        a.rule_id: a
        for a in repo.list_alerts(seeded_engine, shipment_id="shp-0001", status="resolved")
    }
    assert resolved["TEMP_BREACH"].readings_count == 2
    assert "TEMP_ANOMALY" in resolved  # the +2°C spike is also a statistical outlier
    assert repo.list_alerts(seeded_engine, shipment_id="SHP-0001") == []

    open_2 = {a.rule_id for a in repo.list_alerts(seeded_engine, shipment_id="SHP-0002")}
    assert {"PORT_CONGESTION", "TIER2_ESCALATION"} <= open_2


def test_detection_is_idempotent_and_keeps_acknowledgement(seeded_engine):
    def count():
        with session_scope(seeded_engine) as s:
            return s.scalar(select(func.count()).select_from(Alert))

    before = count()
    [breach] = repo.list_alerts(seeded_engine, shipment_id="SHP-0000", rule_id="TEMP_BREACH")
    assert repo.acknowledge(seeded_engine, breach.id).status == "acknowledged"
    run_detection(seeded_engine)
    assert count() == before
    [again] = repo.list_alerts(
        seeded_engine, shipment_id="SHP-0000", rule_id="TEMP_BREACH", status="acknowledged"
    )
    assert again.id == breach.id
    with session_scope(seeded_engine) as s:  # restore for other tests
        s.get(Alert, breach.id).status = "open"


def test_alerts_sorted_most_severe_first(seeded_engine):
    sev = [a.severity for a in repo.list_alerts(seeded_engine, limit=50)]
    rank = {"critical": 0, "high": 1, "medium": 2}
    assert sev == sorted(sev, key=rank.__getitem__)


def test_shipment_status(seeded_engine):
    st = repo.shipment_status(seeded_engine, "shp-0000")
    assert st.cargo_profile == "frozen" and st.allowed_range_c == "≤ -5.0"
    assert st.latest.temperature_c == -1.0
    assert st.temperature_trend.pct_out_of_range == pytest.approx(12.5)
    with pytest.raises(repo.ShipmentNotFound):
        repo.shipment_status(seeded_engine, "SHP-9999")


def test_fleet_summary(seeded_engine):
    fs = repo.fleet_summary(seeded_engine)
    assert fs.shipments == 3 and fs.readings == 384
    assert fs.shipments_by_profile == {"frozen": 2, "fresh": 1}
    assert fs.open_incidents_by_rule["TEMP_BREACH"] == 1
    assert next(iter(fs.open_incidents_by_severity)) == "critical"


def test_sop_reindex_is_incremental(seeded_engine):
    stats = ingest_sop_dir(SOP_DIR, build_embedder(get_settings()), seeded_engine)
    assert stats["added"] == 0 and stats["removed"] == 0 and stats["unchanged"] > 0


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        ("port congestion above 7 at Long Beach", "Route Congestion"),
        ("frozen goods above -5 quarantine", "Temperature Control"),
        ("High Risk delay probability escalation Tier 2", "Risk Classification"),
    ],
)
def test_hybrid_retrieval_top1(seeded_engine, query, expected_section):
    hits = HybridRetriever(seeded_engine, build_embedder(get_settings()), k=3).search(query)
    assert expected_section in hits[0].section
    assert hits[0].citation.startswith("Cold_Chain_Incident_SOP_v2.md")
