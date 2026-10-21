"""Run all three methods in offline mode without spending model/API credits."""
from __future__ import annotations
import csv, time
from pathlib import Path
from features import DEFAULT_DATA, dump_json
from baseline import run as run_baseline
from agent import run as run_agent
from agent_harness import run as run_harness

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def main() -> dict:
    REPORTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    baseline = run_baseline(DEFAULT_DATA, REPORTS / "result_baseline.json")
    agent = run_agent("offline", None, DEFAULT_DATA, REPORTS / "result_agent_offline_demo.json")
    harness = run_harness("offline", None, DEFAULT_DATA, REPORTS / "result_agent_harness_offline_demo.json")
    base_by_id = {r["customer_id"]: r for r in baseline["results"]}
    agent_by_id = {r["customer_id"]: r for r in agent["results"]}
    harness_by_id = {r["customer_id"]: r for r in harness["results"]}
    customers_equal = set(base_by_id) == set(agent_by_id) == set(harness_by_id)
    deterministic_features_equal = customers_equal and all(
        base_by_id[cid]["features"] == agent_by_id[cid]["features"] == harness_by_id[cid]["features"]
        for cid in base_by_id
    )
    accepted_all = harness["accepted_count"] == harness["processed_customers"]
    summary = {
        "study": "ITXN credit activity assessment: three implementation patterns",
        "mode": "offline_demo_only",
        "dataset": baseline["dataset"],
        "programs": [
            {"name": "Python baseline", "file": "result_baseline.json", "customers": len(baseline["results"]), "model_requests": 0},
            {"name": "AI Agent (offline simulation)", "file": "result_agent_offline_demo.json", "customers": len(agent["results"]), "model_requests": 0},
            {"name": "AI Agent + Harness (offline simulation)", "file": "result_agent_harness_offline_demo.json", "customers": len(harness["results"]), "model_requests": 0, "harness_accepted": harness["accepted_count"], "harness_rejected": harness["rejected_needs_review_count"]},
        ],
        "checks": {
            "same_customer_set": customers_equal,
            "same_deterministic_features": deterministic_features_equal,
            "all_harness_demo_outputs_accepted": accepted_all,
        },
        "interpretation": [
            "This demo makes zero model/API calls, so it does not measure a live LLM's reasoning quality, latency, or cost.",
            "The Agent outputs are deterministic stand-ins to check the flow and file contracts.",
            "All three methods describe synthetic transaction activity; none approves or rejects credit.",
            "A live comparison is opt-in per customer and uses at most two model requests per Agent run."
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    dump_json(REPORTS / "comparison_demo.json", summary)

    csv_path = REPORTS / "customer_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "customer_id", "txn_count", "active_days_observed", "amount_sum_observed",
            "reversal_count", "reversal_rate", "response_code_coverage",
            "success_rate_among_known_codes", "baseline_outcome", "agent_outcome", "harness_status"
        ])
        writer.writeheader()
        for cid in sorted(base_by_id):
            b, a, h = base_by_id[cid], agent_by_id[cid], harness_by_id[cid]
            ft = b["features"]
            writer.writerow({
                "customer_id": cid,
                "txn_count": ft["txn_count"],
                "active_days_observed": ft["active_days_observed"],
                "amount_sum_observed": ft["amount_sum_observed"],
                "reversal_count": ft["reversal_count"],
                "reversal_rate": ft["reversal_rate"],
                "response_code_coverage": ft["response_code_coverage"],
                "success_rate_among_known_codes": ft["success_rate_among_known_codes"],
                "baseline_outcome": b["review_outcome"],
                "agent_outcome": a["agent_payload"]["review_outcome"],
                "harness_status": h["harness_status"],
            })
    return summary


if __name__ == "__main__":
    out = main()
    print("Offline case study complete (0 LLM/API calls).")
    print(f"Dataset: {out['dataset']['row_count']} rows / {out['dataset']['customer_count']} synthetic customers")
    print(f"Same customers: {out['checks']['same_customer_set']}")
    print(f"Same deterministic features: {out['checks']['same_deterministic_features']}")
    print(f"Harness accepted: {out['programs'][2]['harness_accepted']}/{out['programs'][2]['customers']}")
    print("Outputs: reports/comparison_demo.json and three result_*.json files")
