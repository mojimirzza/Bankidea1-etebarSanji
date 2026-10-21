"""One-customer live comparison: 1 baseline + 1 Agent response reused by Harness.

The live Agent needs at most two model API requests. The exact same model payload is
then assessed before and after the Harness, avoiding a second LLM run and reducing
cost/variance. Use synthetic data only.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Any
from features import DEFAULT_DATA, dump_json, get_customer_profile, sha256_file
from baseline import baseline_report
from agent_core import run_live_agent
from harness import validate_payload, HarnessValidationError

ROOT = Path(__file__).resolve().parent


def run(customer_id: str, data_path: str | Path = DEFAULT_DATA,
        output_path: str | Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    data_file = Path(data_path)
    profile = get_customer_profile(customer_id, data_file)
    baseline = baseline_report(profile)

    # Only one live Agent call: two inference requests maximum; same payload goes to Harness.
    agent_run = run_live_agent(customer_id, data_file, profile_override=profile)
    payload = agent_run["payload"]

    basic_errors = []
    if not isinstance(payload, dict):
        basic_errors.append("Output is not a JSON object")
    else:
        if payload.get("customer_id") != customer_id:
            basic_errors.append("Customer ID mismatch")
        if payload.get("review_outcome") != "HUMAN_REVIEW_REQUIRED":
            basic_errors.append("Only HUMAN_REVIEW_REQUIRED is permitted")
    basic_status = "ACCEPTED_BASIC_GUARDS" if not basic_errors else "REJECTED_BY_BASIC_GUARDS"

    try:
        validation = validate_payload(payload, profile)
        harness_status = "ACCEPTED"
    except HarnessValidationError as exc:
        validation = {"status": "FAIL", "error": str(exc)}
        harness_status = "REJECTED_NEEDS_REVIEW"

    requested_out = Path(output_path) if output_path else ROOT / "reports" / f"live_comparison_{customer_id}.json"
    report = {
        "study": "ITXN synthetic activity assessment: baseline vs live Agent vs same Agent + Harness",
        "mode": "live_one_customer",
        "customer_id": customer_id,
        "dataset": {
            "file": data_file.name,
            "sha256": sha256_file(data_file),
            "synthetic_only": True,
        },
        "protocol": {
            "model_calls": "One Agent execution (at most 2 API requests). The identical raw Agent payload is reused for the Harness check.",
            "fairness_note": "Agent and Agent+Harness are evaluated on the same response, so differences are attributable to the Harness checks, not a second model sample.",
            "baseline_calls": 0,
            "automatic_retries": 0,
        },
        "baseline": {
            "engine": baseline["engine"],
            "features": baseline["features"],
            "observed_signals": baseline["observed_signals"],
            "missing_evidence": baseline["missing_evidence"],
            "review_outcome": baseline["review_outcome"],
        },
        "agent": {
            "engine": agent_run["engine"],
            "requested_model": agent_run["model_requested"],
            "models_used_reported_by_provider": agent_run["models_used"],
            "api_requests": agent_run["model_requests"],
            "usage_tokens": agent_run["usage"],
            "request_ids": agent_run["request_ids"],
            "runtime_seconds": agent_run["runtime"],
            "basic_guard_status": basic_status,
            "basic_guard_errors": basic_errors,
            "payload": payload,
        },
        "agent_plus_harness": {
            "same_agent_payload_reused": True,
            "harness_status": harness_status,
            "validation": validation,
            "review_outcome": "HUMAN_REVIEW_REQUIRED",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "credit_decision_produced": False,
        "conclusion": [
            "This single-customer run tests a live model's ability to use one read-only tool and produce an evidence summary.",
            "The Harness can reject payload-contract/policy violations; it cannot establish repayment capacity.",
            "One customer is a smoke test, not evidence of general superiority or predictive validity.",
            "Only synthetic data is permitted in this experiment; do not submit real banking/customer data to an external model API.",
        ],
    }
    dump_json(requested_out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer", required=True, help="One synthetic customer, e.g. SYN-CUST-0001")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = run(args.customer, args.data, args.output)
    except Exception as exc:
        raise SystemExit(f"Live comparison stopped safely: {exc}")
    print(f"Live comparison complete: customer={result['customer_id']}")
    print(f"Requested model: {result['agent']['requested_model']}")
    print(f"Actual models reported: {result['agent']['models_used_reported_by_provider']}")
    print(f"Inference requests: {result['agent']['api_requests']} (maximum 2; no retries)")
    print(f"Agent basic guards: {result['agent']['basic_guard_status']}; Harness: {result['agent_plus_harness']['harness_status']}")
    out = args.output or ROOT / "reports" / f"live_comparison_{args.customer}.json"
    print(f"Output: {out}")
