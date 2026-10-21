"""Streamlit front end. A thin client of the API: no DB credentials, no LLM keys,
no business logic here. Anything this UI can do, any other client can do too."""

from __future__ import annotations

import json
import os
import uuid

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("COLDCHAIN_API_URL", "http://localhost:8000")
API_KEY = os.getenv("COLDCHAIN_UI_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="Cold-Chain Ops", page_icon="❄️", layout="wide")


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, headers=HEADERS, timeout=120)


def get(path: str, **params):
    r = client().get(path, params={k: v for k, v in params.items() if v not in (None, "")})
    r.raise_for_status()
    return r.json()


def _detail(resp: httpx.Response) -> str:
    """Human-readable reason from an API error response."""
    if resp.status_code == 503:
        return (
            "The assistant is switched off because no LLM is configured. Add "
            "`COLDCHAIN_LLM_API_KEY` to `.env` and restart with `docker compose up`. "
            "The incident board works without it."
        )
    if resp.status_code == 401:
        return "The API rejected the key. Check `COLDCHAIN_UI_API_KEY`."
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return f"API error {resp.status_code}"


def stream_chat(message: str, thread_id: str):
    """Yield (event, data) pairs from the SSE endpoint."""
    with client().stream(
        "POST", "/v1/chat/stream", json={"message": message, "thread_id": thread_id}
    ) as r:
        if r.is_error:
            r.read()  # load the body so the error detail is available to the caller
            r.raise_for_status()
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[6:])


st.title("❄️ Cold-Chain Operations")
tab_chat, tab_board = st.tabs(["Assistant", "Incident board"])

# --- incident board -----------------------------------------------------------
with tab_board:
    try:
        fs = get("/v1/fleet/summary")
        sev = fs["open_incidents_by_severity"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shipments", fs["shipments"])
        c2.metric("Critical open", sev.get("critical", 0))
        c3.metric("High open", sev.get("high", 0))
        c4.metric("Medium open", sev.get("medium", 0))
        st.caption(f"Data as of {fs['as_of']}")

        f1, f2, f3 = st.columns(3)
        severity = f1.selectbox("Severity", ["", "critical", "high", "medium"])
        rule = f2.selectbox(
            "Rule",
            [
                "",
                "TEMP_BREACH",
                "FREEZE_EXCURSION",
                "PORT_CONGESTION",
                "TIER2_ESCALATION",
                "TEMP_ANOMALY",
            ],
        )
        status = f3.selectbox("Status", ["open", "acknowledged", "resolved", "any"])
        rows = get("/v1/incidents", severity=severity, rule_id=rule, status=status, limit=50)
        if rows:
            df = pd.DataFrame(rows)[
                [
                    "id",
                    "shipment_id",
                    "severity",
                    "rule_id",
                    "peak_value",
                    "readings_count",
                    "started_at",
                    "ended_at",
                    "status",
                    "recommended_action",
                ]
            ]
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No incidents match these filters.")
    except httpx.HTTPError as e:
        st.error(f"API unavailable at {API_URL}: {e}")

# --- assistant ----------------------------------------------------------------
with tab_chat:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
        st.session_state.history = []

    with st.sidebar:
        st.subheader("Try asking")
        for q in [
            "Give me a fleet status briefing.",
            "What are the most critical open incidents and what should we do?",
            "Is SHP-0042 safe to release to the customer?",
            "What does the SOP say about port congestion above 7?",
        ]:
            if st.button(q, use_container_width=True):
                st.session_state.pending = q
        if st.button("New conversation", type="primary", use_container_width=True):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.history = []
            st.rerun()

    for role, content in st.session_state.history:
        st.chat_message(role).markdown(content)

    prompt = st.chat_input("Ask about shipments, incidents or SOPs") or st.session_state.pop(
        "pending", None
    )
    if prompt:
        st.session_state.history.append(("user", prompt))
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            steps = st.status("Working...", expanded=False)
            answer = ""
            try:
                for event, data in stream_chat(prompt, st.session_state.thread_id):
                    if event == "tool_call":
                        steps.write(f"🔧 `{data['name']}` {json.dumps(data['args'])}")
                    elif event == "answer":
                        answer = data["content"]
                    elif event == "error":
                        answer = f"⚠️ {data['detail']}"
                steps.update(label="Done", state="complete")
            except httpx.HTTPStatusError as e:
                steps.update(label="Failed", state="error")
                answer = f"⚠️ {_detail(e.response)}"
            except httpx.HTTPError as e:
                steps.update(label="Failed", state="error")
                answer = f"⚠️ Could not reach the API: {e}"
            st.markdown(answer)
        st.session_state.history.append(("assistant", answer))
