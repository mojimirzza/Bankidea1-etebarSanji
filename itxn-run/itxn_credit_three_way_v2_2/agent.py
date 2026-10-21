"""Program 2: tool-using AI Agent with a minimal safety boundary, no full harness."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from features import DEFAULT_DATA, all_profiles, dump_json
from agent_core import run_agent

ROOT = Path(__file__).resolve().parent


def run(mode: str = "offline", customer_id: str | None = None, data_path: str | Path = DEFAULT_DATA,
        output_path: str | Path | None = None, limit: int | None = None) -> dict:
    profiles, metadata = all_profiles(data_path)
    if mode == "live" and not customer_id:
        raise ValueError("Live mode requires --customer. This avoids accidental multi-call API spend.")
    selected = [p for p in profiles if customer_id is None or p["customer_id"] == customer_id]
    if not selected:
        raise ValueError(f"Unknown synthetic customer id: {customer_id}")
    if limit is not None:
        selected = selected[:limit]
    results = []
    started = time.perf_counter()
    for profile in selected:
        # In live mode each customer costs at most two model requests; errors stop the run with no retries.
        run_result = run_agent(profile["customer_id"], mode=mode, data_path=data_path, profile_override=profile)
        payload = run_result["payload"]
        if payload.get("customer_id") != profile["customer_id"]:
            raise ValueError("Agent returned a mismatched customer id; refusing result")
        if payload.get("review_outcome") != "HUMAN_REVIEW_REQUIRED":
            raise ValueError("Agent attempted a non-human-review outcome; refusing result")
        results.append({
            "customer_id": profile["customer_id"],
            "engine": run_result["engine"],
            "features": profile["features"],
            "agent_payload": payload,
            "model_requests": run_result["model_requests"],
            "model": run_result["model"],
            "model_requested": run_result.get("model_requested"),
            "models_used": run_result.get("models_used", []),
            "usage_tokens": run_result.get("usage"),
            "request_ids": run_result.get("request_ids", []),
            "agent_runtime_seconds": run_result["runtime"],
        })
    report = {
        "program": "ai_agent",
        "mode": mode,
        "dataset": metadata,
        "processed_customers": len(results),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "model_requests_total": sum(item["model_requests"] for item in results),
        "results": results,
        "credit_decision_produced": False,
        "interpretation": "Offline mode is a deterministic simulation, not a live LLM evaluation." if mode == "offline" else "Live model output is not a validated credit decision.",
    }
    target = Path(output_path) if output_path else ROOT / "reports" / ("result_agent_offline_demo.json" if mode == "offline" else "result_agent_live.json")
    dump_json(target, report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["offline", "live"], default="offline")
    ap.add_argument("--customer", help="Required in live mode; e.g. SYN-CUST-0001")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--output")
    ap.add_argument("--limit", type=int, help="Optional cap for offline demo runs")
    args = ap.parse_args()
    try:
        r = run(args.mode, args.customer, args.data, args.output, args.limit)
    except Exception as exc:
        raise SystemExit(f"Agent stopped safely: {exc}")
    print(f"Agent run complete: mode={r['mode']}, customers={r['processed_customers']}, model_requests={r['model_requests_total']}")
    print(f"Output: {args.output or ROOT / 'reports' / ('result_agent_offline_demo.json' if args.mode == 'offline' else 'result_agent_live.json')}")
