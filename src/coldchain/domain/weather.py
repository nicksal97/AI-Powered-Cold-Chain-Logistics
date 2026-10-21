"""Live corridor conditions from Open-Meteo (free, no key).

Replaces the old `8.5 if wind > 10 else 2.5` with a documented scoring based on
WMO weather codes, wind and precipitation, and explains *which* factor drove it.
"""

from __future__ import annotations

import httpx

from coldchain.logging import get_logger
from coldchain.schemas import CorridorConditions

log = get_logger(__name__)

# WMO weather interpretation codes -> (label, risk points)
WMO: dict[int, tuple[str, int]] = {
    0: ("clear", 0),
    1: ("mainly clear", 0),
    2: ("partly cloudy", 0),
    3: ("overcast", 0),
    45: ("fog", 2),
    48: ("depositing rime fog", 3),
    51: ("light drizzle", 1),
    53: ("drizzle", 1),
    55: ("dense drizzle", 1),
    56: ("freezing drizzle", 3),
    57: ("dense freezing drizzle", 3),
    61: ("light rain", 1),
    63: ("rain", 1),
    65: ("heavy rain", 2),
    66: ("freezing rain", 3),
    67: ("heavy freezing rain", 3),
    71: ("light snow", 2),
    73: ("snow", 2),
    75: ("heavy snow", 3),
    77: ("snow grains", 2),
    80: ("rain showers", 1),
    81: ("heavy rain showers", 2),
    82: ("violent rain showers", 3),
    85: ("snow showers", 2),
    86: ("heavy snow showers", 3),
    95: ("thunderstorm", 3),
    96: ("thunderstorm with hail", 3),
    99: ("severe thunderstorm with hail", 3),
}
WIND_HIGH_KMH, WIND_SEVERE_KMH = 40.0, 62.0  # Beaufort 6 / 8
PRECIP_HEAVY_MM = 7.6  # heavy rain, per hour
HEAT_STRESS_C = 30.0  # ambient heat load on reefer units


def score(
    temp: float | None, wind: float | None, precip: float | None, code: int | None
) -> tuple[str, str, list[str]]:
    label, pts = WMO.get(code, ("unknown", 0)) if code is not None else ("unknown", 0)
    factors = [f"weather: {label}"] if pts else []
    if wind is not None and wind >= WIND_SEVERE_KMH:
        pts += 3
        factors.append(f"gale-force wind {wind:.0f} km/h")
    elif wind is not None and wind >= WIND_HIGH_KMH:
        pts += 2
        factors.append(f"strong wind {wind:.0f} km/h")
    if precip is not None and precip >= PRECIP_HEAVY_MM:
        pts += 2
        factors.append(f"heavy precipitation {precip:.1f} mm/h")
    if temp is not None and temp >= HEAT_STRESS_C:
        pts += 1
        factors.append(f"ambient {temp:.0f}°C adds load on reefer cooling")
    risk = "high" if pts >= 3 else "elevated" if pts >= 1 else "normal"
    return label, risk, factors


class WeatherClient:
    def __init__(
        self, base_url: str, timeout_s: float = 6.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            transport=transport or httpx.HTTPTransport(retries=2),
        )

    def corridor(self, latitude: float, longitude: float) -> CorridorConditions:
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError(f"invalid coordinates ({latitude}, {longitude})")
        resp = self._client.get(
            "",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m,precipitation,weather_code",
            },
        )
        resp.raise_for_status()
        cur = resp.json().get("current", {})
        temp, wind = cur.get("temperature_2m"), cur.get("wind_speed_10m")
        precip, code = cur.get("precipitation"), cur.get("weather_code")
        label, risk, factors = score(temp, wind, precip, code)
        return CorridorConditions(
            latitude=latitude,
            longitude=longitude,
            air_temperature_c=temp,
            wind_speed_kmh=wind,
            precipitation_mm=precip,
            weather_code=code,
            conditions=label,
            corridor_risk=risk,
            risk_factors=factors,
        )

    def close(self) -> None:
        self._client.close()
