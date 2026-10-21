from datetime import UTC, datetime, timedelta

import pytest

from coldchain.domain.rules import (
    Reading,
    Severity,
    build_episodes,
    evaluate,
    infer_profile,
    temperature_anomalies,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def r(i=0, temp=2.0, profile="fresh", **kw) -> Reading:
    return Reading(
        reading_id=i,
        shipment_id="SHP-1",
        recorded_at=T0 + timedelta(hours=i),
        temperature_c=temp,
        cargo_profile=profile,
        **kw,
    )


def ids(violations):
    return {v.rule_id for v in violations}


class TestTemperature:
    @pytest.mark.parametrize("temp", [0.0, 2.0, 4.0])
    def test_fresh_in_band_is_clean(self, temp):
        assert evaluate(r(temp=temp)) == []

    def test_fresh_above_band_is_critical_breach(self):
        [v] = evaluate(r(temp=4.1))
        assert v.rule_id == "TEMP_BREACH" and v.severity is Severity.CRITICAL
        assert v.threshold == 4.0

    def test_fresh_below_zero_is_freeze_excursion(self):
        assert ids(evaluate(r(temp=-0.5))) == {"FREEZE_EXCURSION"}

    def test_frozen_cold_is_fine_and_has_no_lower_bound(self):
        assert evaluate(r(temp=-30.0, profile="frozen")) == []

    def test_frozen_above_minus_five_breaches(self):
        [v] = evaluate(r(temp=-4.9, profile="frozen"))
        assert v.rule_id == "TEMP_BREACH" and v.threshold == -5.0
        assert "quarantine" in v.recommended_action

    def test_eta_delay_over_one_hour_adds_diversion(self):
        [v] = evaluate(r(temp=6.0, eta_variation_h=1.5))
        assert "emergency cold-storage" in v.recommended_action
        [v] = evaluate(r(temp=6.0, eta_variation_h=0.5))
        assert "emergency cold-storage" not in v.recommended_action

    def test_profile_inference(self):
        assert infer_profile(-8.0) == "frozen"
        assert infer_profile(2.0) == "fresh"


class TestOperationalRules:
    def test_port_congestion_threshold_is_strict(self):
        assert evaluate(r(port_congestion=7.0)) == []
        assert ids(evaluate(r(port_congestion=7.01))) == {"PORT_CONGESTION"}

    @pytest.mark.parametrize(
        ("risk", "delay", "fires"),
        [
            ("High Risk", 0.66, True),
            ("high risk ", 0.9, True),
            ("High Risk", 0.65, False),
            ("Moderate Risk", 0.99, False),
            ("High Risk", None, False),
        ],
    )
    def test_tier2_escalation(self, risk, delay, fires):
        got = "TIER2_ESCALATION" in ids(evaluate(r(risk_class=risk, delay_probability=delay)))
        assert got is fires


class TestEpisodes:
    def test_consecutive_violations_collapse_and_track_peak(self):
        temps = [2, 5, 7, 6, 2, 2]
        eps = build_episodes([r(i, t) for i, t in enumerate(temps)])
        [ep] = eps
        assert (ep.count, ep.first.reading_id, ep.last.reading_id) == (3, 1, 3)
        assert ep.peak_value == 7 and not ep.is_open

    def test_gap_starts_a_new_episode(self):
        eps = build_episodes([r(i, t) for i, t in enumerate([5, 2, 5])])
        assert len(eps) == 2
        assert [e.is_open for e in eps] == [False, True]

    def test_episode_touching_latest_reading_is_open(self):
        [ep] = build_episodes([r(i, t) for i, t in enumerate([2, 2, 9])])
        assert ep.is_open and ep.count == 1

    def test_independent_rules_overlap(self):
        rs = [r(0, 6, port_congestion=8), r(1, 6, port_congestion=3)]
        eps = {e.rule_id: e for e in build_episodes(rs)}
        assert eps["TEMP_BREACH"].count == 2 and eps["TEMP_BREACH"].is_open
        assert eps["PORT_CONGESTION"].count == 1 and not eps["PORT_CONGESTION"].is_open

    def test_empty(self):
        assert build_episodes([]) == []


class TestAnomalies:
    def test_flags_outlier_against_shipment_baseline(self):
        rs = [r(i, -8.0 + 0.1 * (i % 3), "frozen") for i in range(20)] + [r(20, -6.0, "frozen")]
        flagged = temperature_anomalies(rs)
        assert [x.reading_id for x, _ in flagged] == [20]

    def test_needs_minimum_history(self):
        assert temperature_anomalies([r(i, 10.0 * i) for i in range(5)]) == []

    def test_flat_series_has_no_anomalies(self):
        assert temperature_anomalies([r(i, -8.0) for i in range(30)]) == []
