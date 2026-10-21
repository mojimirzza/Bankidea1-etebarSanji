"""Deterministic, read-only feature extraction for the synthetic ITXN case study."""
from __future__ import annotations
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "synthetic_itxn.csv"
REQUIRED_COLUMNS = {"synthetic_customer_id", "txnamt"}
MISSING_EVIDENCE = ["REPAYMENT_HISTORY", "INCOME_OR_CASHFLOW", "EXISTING_OBLIGATIONS"]
SIGNAL_DEFINITIONS = {
    "TRANSACTION_ACTIVITY_OBSERVED": {
        "description": "At least one transaction row is present for this synthetic customer.",
        "required_fields": ["txn_count"],
    },
    "REVERSALS_OBSERVED": {
        "description": "One or more rows have rev_flag set.",
        "required_fields": ["reversal_count", "reversal_rate"],
    },
    "RESPONSE_CODE_INCOMPLETE": {
        "description": "Some transaction rows have no response code in this dataset.",
        "required_fields": ["response_code_coverage"],
    },
    "AMOUNT_DATA_INCOMPLETE": {
        "description": "Some transaction rows have no parseable amount.",
        "required_fields": ["amount_validity_rate"],
    },
}


def _number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: str | Path = DEFAULT_DATA) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - set(fields)
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("Input dataset is empty")
    for line_no, row in enumerate(rows, start=2):
        if not str(row.get("synthetic_customer_id", "")).strip():
            raise ValueError(f"Blank synthetic_customer_id at CSV line {line_no}")
    return rows, fields


def calculate_features(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compute descriptions of observed activity; never infer repayment capacity."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        customer_id = str(row.get("synthetic_customer_id", "")).strip()
        if not customer_id:
            raise ValueError("Blank synthetic_customer_id encountered")
        groups[customer_id].append(row)

    output: list[dict[str, Any]] = []
    for customer_id in sorted(groups):
        customer_rows = groups[customer_id]
        amounts = [_number(row.get("txnamt")) for row in customer_rows]
        valid_amounts = [amount for amount in amounts if amount is not None]
        known_response_codes = [
            str(row.get("respcode") or "").strip()
            for row in customer_rows
            if str(row.get("respcode") or "").strip()
        ]
        reversal_count = sum(_flag(row.get("rev_flag")) for row in customer_rows)
        active_dates = {
            str(row.get("datein") or "").strip()
            for row in customer_rows
            if str(row.get("datein") or "").strip()
        }
        # Demo convention only: response code 00 is counted as success. Confirm source semantics before using real ITXN.
        success_count = sum(code == "00" for code in known_response_codes)
        count = len(customer_rows)
        features: dict[str, Any] = {
            "synthetic_customer_id": customer_id,
            "txn_count": count,
            "active_days_observed": len(active_dates),
            "amount_sum_observed": round(sum(valid_amounts), 2),
            "amount_avg_observed": round(sum(valid_amounts) / len(valid_amounts), 2) if valid_amounts else None,
            "amount_validity_rate": round(len(valid_amounts) / count, 4),
            "reversal_count": int(reversal_count),
            "reversal_rate": round(reversal_count / count, 4),
            "response_code_coverage": round(len(known_response_codes) / count, 4),
            "success_rate_among_known_codes": round(success_count / len(known_response_codes), 4) if known_response_codes else None,
        }
        output.append(features)
    return output


def make_profile(features: dict[str, Any]) -> dict[str, Any]:
    signals = ["TRANSACTION_ACTIVITY_OBSERVED"]
    if features["reversal_count"] > 0:
        signals.append("REVERSALS_OBSERVED")
    if features["response_code_coverage"] < 1:
        signals.append("RESPONSE_CODE_INCOMPLETE")
    if features["amount_validity_rate"] < 1:
        signals.append("AMOUNT_DATA_INCOMPLETE")
    required_fields = sorted({
        field
        for signal in signals
        for field in SIGNAL_DEFINITIONS[signal]["required_fields"]
    })
    return {
        "customer_id": features["synthetic_customer_id"],
        "features": features,
        "available_signals": signals,
        "signal_definitions": {k: SIGNAL_DEFINITIONS[k] for k in signals},
        "evidence_fields": sorted(set(features) - {"synthetic_customer_id"}),
        "missing_evidence": list(MISSING_EVIDENCE),
        "review_outcome": "HUMAN_REVIEW_REQUIRED",
        "reason": "ITXN transaction activity alone does not establish repayment capacity.",
    }


def all_profiles(path: str | Path = DEFAULT_DATA) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, columns = load_rows(path)
    features = calculate_features(rows)
    profiles = [make_profile(item) for item in features]
    metadata = {
        "dataset_file": Path(path).name,
        "dataset_sha256": sha256_file(Path(path)),
        "row_count": len(rows),
        "customer_count": len(profiles),
        "column_count": len(columns),
        "columns": columns,
        "synthetic_only": True,
    }
    return profiles, metadata


def get_customer_profile(customer_id: str, path: str | Path = DEFAULT_DATA) -> dict[str, Any]:
    profiles, _ = all_profiles(path)
    for profile in profiles:
        if profile["customer_id"] == customer_id:
            return profile
    raise KeyError(f"Unknown synthetic customer id: {customer_id}")


def dump_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
