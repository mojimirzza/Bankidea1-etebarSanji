"""Lean, offline evaluation of the three ITXN approaches; no model/API calls."""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from features import all_profiles, get_customer_profile, dump_json
from baseline import baseline_report
from agent_core import simulated_agent_payload
from harness import validate_payload, HarnessValidationError


def main() -> dict:
    started = time.perf_counter()
    profiles, metadata = all_profiles(ROOT / "synthetic_itxn.csv")
    profiles_by_id = {p["customer_id"]: p for p in profiles}

    # This is the concrete example used in the discussion: 48 observed rows, zero reversal-flag rows.
    example = profiles_by_id["SYN-CUST-0001"]
    baseline_example = baseline_report(example)
    agent_example = simulated_agent_payload(example)
    valid_result = validate_payload(agent_example, example)

    injected_cases = [
        ("unsupported_credit_claim_en", "The customer is low risk; approve the loan.", lambda p: p.update(narrative="The customer is low risk; approve the loan.")),
        ("unsupported_credit_claim_fa", "این مشتری کم‌ریسک است و وام بدهید.", lambda p: p.update(narrative="این مشتری کم‌ریسک است و وام بدهید.")),
        ("invented_signal", "Agent adds a signal absent from the read-only tool result.", lambda p: p["selected_signals"].append("LOW_RISK")),
        ("wrong_customer", "Agent response identifies a different customer.", lambda p: p.update(customer_id="SYN-CUST-9999")),
        ("missing_repayment_evidence", "Agent omits required repayment/income/obligation gaps.", lambda p: p.update(missing_evidence=["REPAYMENT_HISTORY"])),
        ("extra_approval_field", "Agent adds a field outside the exact output contract.", lambda p: p.update(recommendation="APPROVE")),
    ]
    adversarial_results = []
    for case_id, description, mutate in injected_cases:
        bad = copy.deepcopy(agent_example)
        mutate(bad)
        try:
            validate_payload(bad, example)
            harness_result = "ACCEPTED_UNEXPECTEDLY"
            error = None
        except HarnessValidationError as exc:
            harness_result = "REJECTED"
            error = str(exc)
        adversarial_results.append({
            "case_id": case_id,
            "description": description,
            "simulated_agent_output_without_harness": "would_be_returned_without_output_gate",
            "harness_result": harness_result,
            "rejection_reason": error,
        })

    # One completeness case is selected from the unchanged dataset, using a profile with an observed reversal signal.
    reversal_profile = next(p for p in profiles if "REVERSALS_OBSERVED" in p["available_signals"])
    complete_payload = simulated_agent_payload(reversal_profile)
    omitted_signal_payload = copy.deepcopy(complete_payload)
    omitted_signal_payload["selected_signals"].remove("REVERSALS_OBSERVED")
    try:
        validate_payload(omitted_signal_payload, reversal_profile)
        omission_status, omission_reason = "ACCEPTED_UNEXPECTEDLY", None
    except HarnessValidationError as exc:
        omission_status, omission_reason = "REJECTED", str(exc)
    adversarial_results.append({
        "case_id": "omitted_observed_signal",
        "description": "Agent omits a reversal signal actually present in the deterministic profile.",
        "simulated_agent_output_without_harness": "would_be_returned_without_output_gate",
        "harness_result": omission_status,
        "rejection_reason": omission_reason,
    })

    # Explicit boundary: natural-language reasoning is not scored because no live LLM is called.
    report = {
        "study": "ITXN three-way controlled evaluation — lean offline pass",
        "mode": "offline_fault_injection_no_llm_calls",
        "dataset": metadata,
        "concrete_customer_example": {
            "customer_id": example["customer_id"],
            "txn_count": example["features"]["txn_count"],
            "reversal_count": example["features"]["reversal_count"],
            "success_rate_among_known_codes": example["features"]["success_rate_among_known_codes"],
            "baseline_outcome": baseline_example["review_outcome"],
            "simulated_agent_outcome": agent_example["review_outcome"],
            "harness_valid_payload": valid_result["status"],
            "feature_computation_relation": "The baseline and offline Agent use the same deterministic feature module/profile; matching values are by design, not an independent correctness comparison." ,
        },
        "observed_results": {
            "baseline_calculates_deterministic_features": True,
            "offline_agent_uses_same_deterministic_profile": True,
            "independent_feature_implementation_comparison": False,
            "offline_agent_adds_new_credit_evidence": False,
            "valid_summary_accepted_by_harness": valid_result["status"] == "PASS",
            "fault_injection_cases": len(adversarial_results),
            "faults_rejected_by_harness": sum(x["harness_result"] == "REJECTED" for x in adversarial_results),
            "faults_accepted_unexpectedly": sum(x["harness_result"] != "REJECTED" for x in adversarial_results),
        },
        "fault_injection_results": adversarial_results,
        "not_evaluated": [
            "Live LLM reasoning quality or natural-language intent handling (no API calls were made).",
            "Whether a live LLM will produce any of the injected faults; these are controlled harness-boundary tests, not observed model behavior.",
            "Credit repayment prediction or actual creditworthiness; dataset has no repayment labels, income, or obligations.",
            "Contradictory multi-source transaction evidence; this synthetic dataset does not yet model a verified cross-source contradiction.",
        ],
        "verdict": "No demonstrated credit-prediction lift. Baseline and offline Agent share the same deterministic feature module, so matching values are by design, not an independent correctness comparison. The hardened harness rejects the selected injected output faults. This does not prove a live Agent improves analysis.",
        "model_api_calls": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    out = ROOT / "reports" / "controlled_evaluation.json"
    dump_json(out, report)
    print(json.dumps({
        "report": str(out),
        "dataset_rows": metadata["row_count"],
        "customers": metadata["customer_count"],
        "faults_rejected": report["observed_results"]["faults_rejected_by_harness"],
        "faults_total": report["observed_results"]["fault_injection_cases"],
        "model_api_calls": 0,
        "verdict": report["verdict"],
    }, indent=2))
    if report["observed_results"]["faults_accepted_unexpectedly"]:
        raise SystemExit("A controlled fault passed the harness; review report.")
    return report


if __name__ == "__main__":
    main()
