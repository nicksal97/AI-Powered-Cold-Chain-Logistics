import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from coldchain.api.container import build_container
from coldchain.api.main import create_app
from coldchain.config import Settings, get_settings
from coldchain.domain.weather import WeatherClient
from tests.conftest import ScriptedChatModel, tool_call

pytestmark = pytest.mark.integration


def fake_weather() -> WeatherClient:
    return WeatherClient(
        "https://weather.test/v1/forecast",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 31.0,
                        "wind_speed_10m": 12.0,
                        "precipitation": 0.0,
                        "weather_code": 0,
                    }
                },
            )
        ),
    )


def make_client(engine, script, settings: Settings | None = None):
    settings = settings or get_settings()
    llm = ScriptedChatModel(script=script, seen=[])

    def factory(s):
        return build_container(s, llm=llm, engine=engine, weather=fake_weather())

    return TestClient(create_app(settings, factory)), llm


@pytest.fixture
def client(seeded_engine):
    c, _ = make_client(seeded_engine, [])
    with c:
        yield c


def test_health_and_metrics(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["database"] is True
    r = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert "coldchain_http_requests_total" in client.get("/metrics").text


def test_ops_endpoints(client):
    assert client.get("/v1/fleet/summary").json()["shipments"] == 3
    rows = client.get("/v1/incidents", params={"severity": "critical"}).json()
    assert [r["shipment_id"] for r in rows] == ["SHP-0000"]
    assert client.get("/v1/shipments/SHP-0002").json()["cargo_profile"] == "fresh"
    assert client.get("/v1/shipments/NOPE").status_code == 404
    assert client.get("/v1/incidents", params={"limit": 500}).status_code == 422
    hits = client.get("/v1/sop/search", params={"q": "port congestion"}).json()
    assert "Route Congestion" in hits[0]["citation"]


def test_acknowledge_flow(client):
    [inc] = client.get(
        "/v1/incidents", params={"shipment_id": "SHP-0002", "rule_id": "PORT_CONGESTION"}
    ).json()
    r = client.post(f"/v1/incidents/{inc['id']}/ack")
    assert r.status_code == 200 and r.json()["status"] == "acknowledged"
    assert client.post("/v1/incidents/999999/ack").status_code == 404
    client.app.state.container.engine  # noqa: B018
    from coldchain.db.models import Alert
    from coldchain.db.session import session_scope

    with session_scope(client.app.state.container.engine) as s:
        s.get(Alert, inc["id"]).status = "open"


def test_api_key_enforced(seeded_engine):
    s = get_settings().model_copy(update={"api_keys": ["secret"]})
    c, _ = make_client(seeded_engine, [], s)
    with c:
        assert c.get("/v1/fleet/summary").status_code == 401
        assert c.get("/v1/fleet/summary", headers={"X-API-Key": "nope"}).status_code == 401
        assert c.get("/v1/fleet/summary", headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/healthz").status_code == 200  # probes stay open


def test_chat_runs_real_tools_and_keeps_thread_memory(seeded_engine):
    script = [
        tool_call("get_shipment_status", {"shipment_id": "SHP-0000"}, "c1"),
        tool_call("search_sop", {"query": "frozen goods breach quarantine"}, "c2"),
        AIMessage(content="**Summary** SHP-0000 is in an open frozen-chain breach."),
        AIMessage(content="Still SHP-0000."),
    ]
    c, llm = make_client(seeded_engine, script)
    with c:
        r = c.post("/v1/chat", json={"message": "Is SHP-0000 ok?"})
        body = r.json()
        assert r.status_code == 200, body
        assert [t["name"] for t in body["tool_calls"]] == ["get_shipment_status", "search_sop"]
        assert body["answer"].startswith("**Summary**")

        # the LLM actually received real tool output from Postgres
        tool_msgs = [m for m in llm.seen[2] if isinstance(m, ToolMessage)]
        status = json.loads(tool_msgs[0].content)
        assert status["latest"]["temperature_c"] == -1.0
        assert "Temperature Control" in tool_msgs[1].content

        r2 = c.post("/v1/chat", json={"message": "and now?", "thread_id": body["thread_id"]})
        assert r2.json()["tool_calls"] == []  # only this turn's calls are reported
        assert any("Is SHP-0000 ok?" in str(m.content) for m in llm.seen[-1])  # memory


def test_tool_errors_are_returned_to_the_model_not_raised(seeded_engine):
    script = [
        tool_call("get_shipment_status", {"shipment_id": "SHP-9999"}),
        AIMessage(content="I could not find that shipment."),
    ]
    c, llm = make_client(seeded_engine, script)
    with c:
        assert c.post("/v1/chat", json={"message": "SHP-9999?"}).status_code == 200
    err = json.loads(next(m for m in llm.seen[1] if isinstance(m, ToolMessage)).content)
    assert err["error"] == "ShipmentNotFound"


def test_step_budget(seeded_engine):
    loop = [tool_call("get_fleet_summary", {}, f"c{i}") for i in range(50)]
    s = get_settings().model_copy(update={"agent_max_steps": 2})
    c, _ = make_client(seeded_engine, loop, s)
    with c:
        assert c.post("/v1/chat", json={"message": "loop"}).status_code == 422


def test_stream_emits_tool_and_answer_events(seeded_engine):
    script = [
        tool_call("get_corridor_conditions", {"latitude": 33.7, "longitude": -118.2}),
        AIMessage(content="Corridor is elevated due to heat."),
    ]
    c, _ = make_client(seeded_engine, script)
    with c, c.stream("POST", "/v1/chat/stream", json={"message": "weather?"}) as r:
        events = [ln[7:] for ln in r.iter_lines() if ln.startswith("event: ")]
    assert events == ["thread", "tool_call", "tool_result", "answer", "done"]


def test_request_validation(client):
    assert client.post("/v1/chat", json={"message": ""}).status_code == 422
    assert client.post("/v1/chat", json={"message": "x", "thread_id": "bad id!"}).status_code == 422


def test_chat_returns_503_without_llm_but_ops_still_work(seeded_engine):
    s = get_settings().model_copy(update={"llm_api_key": None, "llm_provider": "openai"})

    def factory(st):
        return build_container(st, engine=seeded_engine, weather=fake_weather())

    with TestClient(create_app(s, factory)) as c:
        assert c.post("/v1/chat", json={"message": "hi"}).status_code == 503
        assert c.get("/v1/fleet/summary").status_code == 200
