import math

import httpx
import pytest

from coldchain.config import Settings
from coldchain.domain.weather import WeatherClient, score
from coldchain.rag.chunking import chunk_markdown
from coldchain.rag.embeddings import HashEmbedder

DOC = """# Ops SOP
intro line
## 1. Temperature
* fresh 0-4C
### 1.1 Mitigation
restart the unit
## 2. Ports
divert above 7
"""


class TestChunking:
    def test_sections_carry_full_heading_path(self):
        chunks = chunk_markdown(DOC, "sop.md")
        assert [c.section for c in chunks] == [
            "Ops SOP",
            "Ops SOP > 1. Temperature",
            "Ops SOP > 1. Temperature > 1.1 Mitigation",
            "Ops SOP > 2. Ports",
        ]

    def test_long_sections_split_with_stable_hashes(self):
        body = "\n".join(f"* rule number {i} " + "x" * 80 for i in range(40))
        a = chunk_markdown(f"# S\n{body}", "d.md", max_chars=500)
        b = chunk_markdown(f"# S\n{body}", "d.md", max_chars=500)
        assert len(a) > 1 and all(len(c.content) <= 700 for c in a)
        assert [c.content_hash for c in a] == [c.content_hash for c in b]


class TestHashEmbedder:
    def test_deterministic_and_normalised(self):
        e = HashEmbedder(64)
        v1, v2 = e.embed_query("port congestion"), e.embed_query("port congestion")
        assert v1 == v2 and len(v1) == 64
        assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-9)

    def test_overlap_scores_higher(self):
        e = HashEmbedder(512)
        q = e.embed_query("port congestion diversion")
        near = e.embed_query("divert when port congestion is high")
        far = e.embed_query("reefer temperature sensor calibration")
        cos = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
        assert cos(q, near) > cos(q, far)


class TestWeather:
    def test_scoring(self):
        assert score(15, 10, 0, 0)[1] == "normal"
        assert score(15, 45, 0, 61)[1] == "high"  # strong wind (2) + rain (1)
        label, risk, factors = score(35, 5, 0, 95)
        assert (label, risk) == ("thunderstorm", "high")
        assert any("reefer" in f for f in factors)

    def test_client_parses_open_meteo(self):
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.params["latitude"] == "33.7"
            return httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 21.0,
                        "wind_speed_10m": 70.0,
                        "precipitation": 0.0,
                        "weather_code": 3,
                    }
                },
            )

        c = WeatherClient(
            "https://example.test/v1/forecast", transport=httpx.MockTransport(handler)
        )
        out = c.corridor(33.7, -118.2)
        assert out.corridor_risk == "high" and out.conditions == "overcast"

    def test_rejects_bad_coordinates(self):
        with pytest.raises(ValueError):
            WeatherClient("https://x.test").corridor(123, 0)


def test_prod_requires_api_keys():
    with pytest.raises(ValueError, match="API_KEYS"):
        Settings(env="prod", api_keys=[])
    assert Settings(env="prod", api_keys=["k"]).api_keys == ["k"]
