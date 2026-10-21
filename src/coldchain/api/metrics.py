from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "coldchain_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "coldchain_http_request_seconds", "HTTP request latency", ["method", "route"]
)
AGENT_RUNS = Counter("coldchain_agent_runs_total", "Agent runs", ["outcome"])
TOOL_CALLS = Counter("coldchain_agent_tool_calls_total", "Tool calls by the agent", ["tool"])
AGENT_LATENCY = Histogram(
    "coldchain_agent_seconds",
    "End-to-end agent latency",
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80),
)
