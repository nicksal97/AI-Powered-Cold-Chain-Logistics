"""SOP rules as code.

The SOP (data/policy/Cold_Chain_Incident_SOP_v2.md) defines hard thresholds. Those
are compliance rules, so they are evaluated deterministically here, not left to the
LLM to "notice" in a table of numbers. The agent's job is to explain the alerts and
pull the matching procedure text, not to be the detector.

Each rule is a small, pure, independently testable object. Adding a rule means
adding one entry to DEFAULT_RULES.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2}


@dataclass(frozen=True, slots=True)
class Reading:
    reading_id: int
    shipment_id: str
    recorded_at: datetime
    temperature_c: float
    cargo_profile: str = "fresh"
    risk_class: str | None = None
    delay_probability: float | None = None
    port_congestion: float | None = None
    eta_variation_h: float | None = None


@dataclass(frozen=True, slots=True)
class Violation:
    rule_id: str
    severity: Severity
    message: str
    recommended_action: str
    sop_reference: str
    observed_value: float | None
    threshold: float | None


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    sop_reference: str
    check: Callable[[Reading], Violation | None]


# --- thresholds straight from SOP v2.5 ---------------------------------------
@dataclass(frozen=True, slots=True)
class TempBand:
    min_c: float | None
    max_c: float
    breach_action: str


PROFILES: dict[str, TempBand] = {
    "fresh": TempBand(
        0.0,
        4.0,
        "Contact driver to restart auxiliary cooling unit.",
    ),
    "frozen": TempBand(
        None,
        -5.0,
        "Contact driver to verify reefer setpoint and door seals. If the breach persists for "
        "3+ consecutive readings, quarantine on arrival pending QA inspection.",
    ),
}
FROZEN_PROFILE_MEDIAN_C = -2.0  # trips whose median temperature is below this are frozen loads
PORT_CONGESTION_MAX = 7.0
ESCALATION_DELAY_PROB = 0.65
DIVERT_ETA_DELAY_H = 1.0


def infer_profile(median_temperature_c: float) -> str:
    """The source data has no product type, so the profile is inferred from the trip's
    median temperature. In production this comes from the booking / TMS record."""
    return "frozen" if median_temperature_c < FROZEN_PROFILE_MEDIAN_C else "fresh"


def _band(r: Reading) -> TempBand:
    return PROFILES.get(r.cargo_profile, PROFILES["fresh"])


def _temp_breach(r: Reading) -> Violation | None:
    band = _band(r)
    if r.temperature_c <= band.max_c:
        return None
    action = band.breach_action
    if r.eta_variation_h is not None and r.eta_variation_h > DIVERT_ETA_DELAY_H:
        action += " ETA delay exceeds 1h: divert to nearest emergency cold-storage facility."
    return Violation(
        rule_id="TEMP_BREACH",
        severity=Severity.CRITICAL,
        message=(
            f"{r.cargo_profile.title()} cargo at {r.temperature_c:.1f}°C, above the "
            f"{band.max_c:.1f}°C limit."
        ),
        recommended_action=action,
        sop_reference="SOP v2.5 §1 Temperature Control",
        observed_value=r.temperature_c,
        threshold=band.max_c,
    )


def _freeze_excursion(r: Reading) -> Violation | None:
    band = _band(r)
    if band.min_c is None or r.temperature_c >= band.min_c:
        return None
    return Violation(
        rule_id="FREEZE_EXCURSION",
        severity=Severity.HIGH,
        message=(
            f"{r.cargo_profile.title()} cargo at {r.temperature_c:.1f}°C, below the "
            f"{band.min_c:.1f}°C lower bound."
        ),
        recommended_action=(
            "Check reefer setpoint and thermostat; inspect perishables for freeze damage."
        ),
        sop_reference="SOP v2.5 §1 Temperature Control",
        observed_value=r.temperature_c,
        threshold=band.min_c,
    )


def _port_congestion(r: Reading) -> Violation | None:
    if r.port_congestion is None or r.port_congestion <= PORT_CONGESTION_MAX:
        return None
    return Violation(
        rule_id="PORT_CONGESTION",
        severity=Severity.HIGH,
        message=f"Port congestion {r.port_congestion:.1f}, above the {PORT_CONGESTION_MAX} limit.",
        recommended_action=(
            "Suspend standard routing. Divert to Inland Empire Overflow Depot "
            "(San Bernardino) for cross-docking. Do not hold freight at port."
        ),
        sop_reference="SOP v2.5 §2 Route Congestion",
        observed_value=r.port_congestion,
        threshold=PORT_CONGESTION_MAX,
    )


def _escalation(r: Reading) -> Violation | None:
    if (
        (r.risk_class or "").strip().lower() != "high risk"
        or r.delay_probability is None
        or r.delay_probability <= ESCALATION_DELAY_PROB
    ):
        return None
    return Violation(
        rule_id="TIER2_ESCALATION",
        severity=Severity.MEDIUM,
        message=(
            f"High Risk shipment with delay probability {r.delay_probability:.2f} "
            f"(> {ESCALATION_DELAY_PROB})."
        ),
        recommended_action="Escalate to Tier 2 Logistics Manager.",
        sop_reference="SOP v2.5 §3 Risk Classification",
        observed_value=r.delay_probability,
        threshold=ESCALATION_DELAY_PROB,
    )


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("TEMP_BREACH", "SOP v2.5 §1", _temp_breach),
    Rule("FREEZE_EXCURSION", "SOP v2.5 §1", _freeze_excursion),
    Rule("PORT_CONGESTION", "SOP v2.5 §2", _port_congestion),
    Rule("TIER2_ESCALATION", "SOP v2.5 §3", _escalation),
)


def evaluate(reading: Reading, rules: Sequence[Rule] = DEFAULT_RULES) -> list[Violation]:
    return [v for rule in rules if (v := rule.check(reading)) is not None]


# --- statistical layer --------------------------------------------------------
def temperature_anomalies(
    readings: Sequence[Reading], z_threshold: float = 3.5, min_points: int = 10
) -> list[tuple[Reading, float]]:
    """Flag readings whose temperature is unusual *for that shipment*.

    Uses the modified z-score (median / MAD), which is robust to the very outliers
    it is trying to find. Catches sensor faults and sudden jumps that may still sit
    inside the SOP band, which a fixed threshold cannot see.
    """
    if len(readings) < min_points:
        return []
    temps = [r.temperature_c for r in readings]
    med = statistics.median(temps)
    mad = statistics.median(abs(t - med) for t in temps)
    if mad == 0:
        return []
    out = []
    for r in readings:
        z = 0.6745 * (r.temperature_c - med) / mad
        if abs(z) > z_threshold:
            out.append((r, round(z, 2)))
    return out


# --- episodes -------------------------------------------------------------------
@dataclass(slots=True)
class Episode:
    shipment_id: str
    rule_id: str
    severity: Severity
    sop_reference: str
    threshold: float | None
    first: Reading
    last: Reading
    peak: Violation
    count: int = 1
    is_open: bool = False

    @property
    def peak_value(self) -> float | None:
        return self.peak.observed_value

    def summary(self) -> str:
        span = "1 reading" if self.count == 1 else f"{self.count} consecutive readings"
        return f"{self.peak.message} Excursion over {span}."


def _worse(a: Violation, b: Violation) -> Violation:
    """Keep the violation furthest past its threshold."""
    if a.observed_value is None or a.threshold is None:
        return b
    if b.observed_value is None:
        return a
    return b if abs(b.observed_value - a.threshold) > abs(a.observed_value - a.threshold) else a


def build_episodes(
    readings: Sequence[Reading],
    rules: Sequence[Rule] = DEFAULT_RULES,
    extra: dict[int, list[Violation]] | None = None,
) -> list[Episode]:
    """Collapse one shipment's chronologically ordered readings into rule episodes.

    `extra` injects violations computed elsewhere (e.g. the statistical anomaly layer),
    keyed by reading_id, so they get the same episode treatment.
    """
    episodes: list[Episode] = []
    active: dict[str, Episode] = {}
    for r in readings:
        found = {v.rule_id: v for v in evaluate(r, rules)}
        for v in (extra or {}).get(r.reading_id, []):
            found[v.rule_id] = v
        for rule_id in list(active):
            if rule_id not in found:
                active.pop(rule_id)
        for rule_id, v in found.items():
            ep = active.get(rule_id)
            if ep is None:
                ep = Episode(
                    r.shipment_id,
                    rule_id,
                    v.severity,
                    v.sop_reference,
                    v.threshold,
                    first=r,
                    last=r,
                    peak=v,
                )
                active[rule_id] = ep
                episodes.append(ep)
            else:
                ep.last, ep.count, ep.peak = r, ep.count + 1, _worse(ep.peak, v)
    if readings:
        latest_id = readings[-1].reading_id
        for ep in episodes:
            ep.is_open = ep.last.reading_id == latest_id
    return episodes
