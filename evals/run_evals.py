"""Offline evaluation.

  python evals/run_evals.py retrieval [--min-hit1 0.8]
      Hit@1, Hit@3 and MRR for the SOP retriever over evals/retrieval_golden.json.
      No API keys needed with COLDCHAIN_EMBEDDING_PROVIDER=hash; runs in CI as a gate.

  python evals/run_evals.py agent [--min-pass 0.8]
      Runs each case in evals/agent_golden.json through the real agent and checks
      tool selection (required tools called, nothing called when it shouldn't be) and
      grounding (required facts appear in the answer). Needs an LLM key and seeded DB.

Both write a JSON report to evals/reports/ so results can be tracked across prompt,
model and retriever changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from coldchain.config import get_settings

HERE = Path(__file__).resolve().parent


def _report(name: str, payload: dict) -> Path:
    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"{name}-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def eval_retrieval(min_hit1: float) -> bool:
    from coldchain.db.session import get_engine
    from coldchain.rag.embeddings import build_embedder
    from coldchain.rag.retriever import HybridRetriever

    s = get_settings()
    retriever = HybridRetriever(get_engine(), build_embedder(s), k=3)
    cases = json.loads((HERE / "retrieval_golden.json").read_text())
    rows, rr = [], []
    for c in cases:
        hits = retriever.search(c["query"])
        rank = next((i + 1 for i, h in enumerate(hits) if c["expected_section"] in h.section), None)
        rr.append(1 / rank if rank else 0.0)
        rows.append({**c, "rank": rank, "top": hits[0].section if hits else None})
    n = len(cases)
    metrics = {
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        "hit@3": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(rr) / n,
        "embedder": s.embedding_provider,
        "cases": n,
    }
    path = _report("retrieval", {"metrics": metrics, "cases": rows})
    print(json.dumps(metrics, indent=2))
    for r in rows:
        if r["rank"] != 1:
            print(f"  miss rank={r['rank']}: {r['query']!r} -> {r['top']}")
    print(f"report: {path}")
    return metrics["hit@1"] >= min_hit1


def eval_agent(min_pass: float) -> bool:
    from coldchain.api.container import build_container

    s = get_settings().model_copy(update={"checkpointer": "memory"})
    c = build_container(s)
    if c.agent is None:
        print("No LLM configured; set COLDCHAIN_LLM_API_KEY.")
        return False
    cases = json.loads((HERE / "agent_golden.json").read_text())
    results = []
    for case in cases:
        t0 = time.perf_counter()
        out = c.agent.invoke(
            {"messages": [HumanMessage(case["question"])]},
            {"configurable": {"thread_id": f"eval-{case['id']}"}, "recursion_limit": 25},
        )
        calls = [
            tc["name"] for m in out["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls
        ]
        answer = str(out["messages"][-1].content)
        checks = {
            "required_tools": all(t in calls for t in case["must_call"]),
            "grounded": any(k.lower() in answer.lower() for k in case["must_mention_any"]),
        }
        if case.get("must_not_call_any"):
            checks["no_tools"] = not calls
        results.append(
            {
                "id": case["id"],
                "passed": all(checks.values()),
                "checks": checks,
                "tools": calls,
                "seconds": round(time.perf_counter() - t0, 1),
                "answer": answer,
            }
        )
        mark = "PASS" if results[-1]["passed"] else "FAIL"
        print(f"{mark} {case['id']:<18} tools={calls} {checks}")
    rate = sum(r["passed"] for r in results) / len(results)
    path = _report(
        "agent", {"model": f"{s.llm_provider}/{s.llm_model}", "pass_rate": rate, "results": results}
    )
    print(f"pass rate {rate:.0%}  report: {path}")
    c.close()
    return rate >= min_pass


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="suite", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--min-hit1", type=float, default=0.8)
    a = sub.add_parser("agent")
    a.add_argument("--min-pass", type=float, default=0.8)
    args = p.parse_args()
    ok = eval_retrieval(args.min_hit1) if args.suite == "retrieval" else eval_agent(args.min_pass)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
