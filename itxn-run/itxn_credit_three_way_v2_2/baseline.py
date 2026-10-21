"""Program 1: deterministic Python baseline; no LLM and no agent."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from features import DEFAULT_DATA, all_profiles, dump_json

ROOT = Path(__file__).resolve().parent


def baseline_report(profile: dict) -> dict:
    f = profile["features"]
    findings = ["TRANSACTION_ACTIVITY_OBSERVED"]
    if "REVERSALS_OBSERVED" in profile["available_signals"]:
        findings.append("REVERSALS_OBSERVED")
    if "RESPONSE_CODE_INCOMPLETE" in profile["available_signals"]:
        findings.append("RESPONSE_CODE_INCOMPLETE")
    if "AMOUNT_DATA_INCOMPLETE" in profile["available_signals"]:
        findings.append("AMOUNT_DATA_INCOMPLETE")
    return {
        "customer_id": profile["customer_id"],
        "engine": "deterministic_python_baseline",
        "features": f,
        "observed_signals": findings,
        "missing_evidence": profile["missing_evidence"],
        "assessment_summary": "Observed switch activity is summarized; it is not evidence of repayment capacity.",
        "review_outcome": "HUMAN_REVIEW_REQUIRED",
    }


def run(data_path: str | Path = DEFAULT_DATA, output_path: str | Path = ROOT / "reports/result_baseline.json") -> dict:
    started = time.perf_counter()
    profiles, metadata = all_profiles(data_path)
    report = {
        "program": "python_baseline",
        "mode": "deterministic",
        "dataset": metadata,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "results": [baseline_report(p) for p in profiles],
        "credit_decision_produced": False,
    }
    dump_json(output_path, report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--output", default=str(ROOT / "reports/result_baseline.json"))
    args = ap.parse_args()
    r = run(args.data, args.output)
    print(f"Baseline complete: {r['dataset']['row_count']} rows, {r['dataset']['customer_count']} synthetic customers")
    print(f"Output: {args.output}")
