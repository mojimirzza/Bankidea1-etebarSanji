"""Strict deterministic validation controls for the bounded ITXN evidence-summary agent."""
from __future__ import annotations
import hashlib, json, re
from typing import Any

REQUIRED_MISSING = {"REPAYMENT_HISTORY", "INCOME_OR_CASHFLOW", "EXISTING_OBLIGATIONS"}
REQUIRED_KEYS = {
    "customer_id", "narrative", "selected_signals", "evidence_fields",
    "missing_evidence", "review_outcome"
}
# Intentionally conservative: the summary is not allowed to make credit decisions.
FORBIDDEN_CREDIT_CLAIMS = re.compile(
    r"\b(approved|approve|rejected|reject|eligible|creditworthy|low risk|high risk|"
    r"good borrower|bad borrower|grant (the )?loan|deny (the )?loan|repayment capacity confirmed)\b"
    r"|(?:کم\s*ریسک|ریسک\s*پایین|پرریسک|ریسک\s*بالا|واجد\s*شرایط\s*(?:اعتبار|دریافت\s*وام)?|"
    r"تأیید\s*(?:اعتبار|وام)|تایید\s*(?:اعتبار|وام)|وام\s*بدهید|اعتبار\s*بدهید|"
    r"توان\s*بازپرداخت\s*(?:دارد|تأیید\s*شده))",
    re.I,
)


class HarnessValidationError(ValueError):
    """A generated report violated its output, evidence, or policy contract."""


def _require_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise HarnessValidationError(f"{field} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise HarnessValidationError(f"{field} must not contain duplicates")
    return value


def validate_payload(payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete evidence summary. This does not validate creditworthiness."""
    if not isinstance(payload, dict):
        raise HarnessValidationError("Output must be a JSON object")
    missing_keys = REQUIRED_KEYS - set(payload)
    extra_keys = set(payload) - REQUIRED_KEYS
    if missing_keys:
        raise HarnessValidationError("Missing output keys: " + ", ".join(sorted(missing_keys)))
    if extra_keys:
        raise HarnessValidationError("Unexpected output keys: " + ", ".join(sorted(extra_keys)))
    if payload["customer_id"] != profile["customer_id"]:
        raise HarnessValidationError("Customer ID mismatch")
    if payload["review_outcome"] != "HUMAN_REVIEW_REQUIRED":
        raise HarnessValidationError("Unsafe outcome: only HUMAN_REVIEW_REQUIRED is permitted")
    narrative = payload["narrative"]
    if not isinstance(narrative, str) or not narrative.strip():
        raise HarnessValidationError("Narrative must be a non-empty string")
    if re.search(r"\d", narrative):
        raise HarnessValidationError("Narrative contains digits; keep quantitative facts in deterministic features")
    if FORBIDDEN_CREDIT_CLAIMS.search(narrative):
        raise HarnessValidationError("Narrative contains a prohibited credit conclusion")

    selected_list = _require_string_list(payload["selected_signals"], "selected_signals")
    selected = set(selected_list)
    allowed_signals = set(profile["available_signals"])
    if not selected <= allowed_signals:
        raise HarnessValidationError("Unsupported signal selected: " + ", ".join(sorted(selected - allowed_signals)))
    # This task asks for a complete activity summary; silently omitting any tool-observed signal is invalid.
    if selected != allowed_signals:
        omitted = allowed_signals - selected
        raise HarnessValidationError("Observed signal(s) omitted: " + ", ".join(sorted(omitted)))

    evidence_list = _require_string_list(payload["evidence_fields"], "evidence_fields")
    citations = set(evidence_list)
    allowed_fields = set(profile["evidence_fields"])
    if not citations <= allowed_fields:
        raise HarnessValidationError("Unknown evidence field cited: " + ", ".join(sorted(citations - allowed_fields)))
    required_citations = {
        field for code in selected
        for field in profile["signal_definitions"][code]["required_fields"]
    }
    if not required_citations <= citations:
        raise HarnessValidationError("Evidence citations do not support all selected signals")

    missing_list = _require_string_list(payload["missing_evidence"], "missing_evidence")
    missing_evidence = set(missing_list)
    if missing_evidence != REQUIRED_MISSING:
        absent = REQUIRED_MISSING - missing_evidence
        unexpected = missing_evidence - REQUIRED_MISSING
        details = []
        if absent:
            details.append("omitted=" + ",".join(sorted(absent)))
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        raise HarnessValidationError("Missing-evidence contract mismatch: " + "; ".join(details))
    return {
        "status": "PASS",
        "checks": [
            "exact output schema", "customer identity", "human review only",
            "all observed signals included", "signals supported by tool output",
            "evidence field citations", "required missing evidence present",
            "no numeric claims in narrative", "no prohibited credit conclusion",
        ],
    }


def audit_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
