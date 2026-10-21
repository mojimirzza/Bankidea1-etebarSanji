"""Bounded tool-using agent. Offline mode is a labelled simulation; live mode calls an OpenAI-compatible API."""
from __future__ import annotations
import json, os, re, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any
from features import DEFAULT_DATA, get_customer_profile

TOOL_NAME = "get_customer_profile"
TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Read-only tool. Return the precomputed synthetic ITXN activity profile for exactly the requested synthetic customer.",
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string", "description": "Exact synthetic customer identifier from the supplied task."}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
}]
SYSTEM_PROMPT = """You are an evidence summarization agent in a synthetic banking case study.
You MUST call get_customer_profile exactly once for the customer named by the user. It is read-only.
After receiving the tool result, return only a JSON object with keys:
customer_id (string), narrative (string), selected_signals (array of signal-code strings),
evidence_fields (array of field-name strings), missing_evidence (array of required missing-evidence codes),
review_outcome (must be HUMAN_REVIEW_REQUIRED).
Include every signal returned by the tool in selected_signals, and do not add any other signal. Cite source fields through evidence_fields. Return exactly the required keys and no extras. Never invent metrics or claim a customer's repayment capacity, creditworthiness, eligibility, or risk grade. Do not recommend approval/rejection or an amount/limit. The ITXN data contains transaction activity only. Narrative must be concise, cautious, and contain no digits. List all three missing-evidence codes supplied by the tool. Do not include a markdown code fence."""


def simulated_agent_payload(profile: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in for local demos; it is NOT an LLM call."""
    signals = list(profile["available_signals"])
    fields = sorted({f for s in signals for f in profile["signal_definitions"][s]["required_fields"]})
    return {
        "customer_id": profile["customer_id"],
        "narrative": "Observed transaction activity can be summarized, but this dataset cannot establish repayment capacity. Any decision requires independent evidence and human review.",
        "selected_signals": signals,
        "evidence_fields": fields,
        "missing_evidence": list(profile["missing_evidence"]),
        "review_outcome": "HUMAN_REVIEW_REQUIRED",
    }


def _api_base() -> str:
    return os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


def _requested_model() -> str:
    """Default OpenRouter to its documented free router; other providers must be explicit."""
    configured = os.environ.get("LLM_MODEL", "").strip()
    if configured:
        return configured
    if "openrouter.ai" in _api_base().lower():
        return "openrouter/free"
    raise RuntimeError("Set LLM_MODEL for this provider. No API call was made.")


def _api_key() -> str:
    # Prefer an explicitly generic key, then the key that matches the configured endpoint.
    key = os.environ.get("LLM_API_KEY", "").strip()
    if key:
        return key
    base = _api_base().lower()
    if "openrouter.ai" in base:
        return os.environ.get("OPENROUTER_API_KEY", "").strip()
    if "api.openai.com" in base:
        return os.environ.get("OPENAI_API_KEY", "").strip()
    return ""


def _post_chat(payload: dict[str, Any], timeout_s: int = 45) -> dict[str, Any]:
    key = _api_key()
    model = _requested_model()
    if not key:
        raise RuntimeError("Live mode requires a key for the configured provider. No API call was made.")
    payload = {**payload, "model": model, "temperature": 0, "max_tokens": 700}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if "openrouter.ai" in _api_base().lower():
        headers["HTTP-Referer"] = os.environ.get("OPENROUTER_SITE_URL", "https://localhost/itxn-credit-case-study")
        headers["X-Title"] = os.environ.get("OPENROUTER_APP_NAME", "ITXN Credit Case Study")
    request = urllib.request.Request(
        _api_base() + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Model API returned HTTP {exc.code}: {body}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Model API connection failed: {exc.reason}") from None


def _parse_json_content(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Model output must be a JSON object")
    return obj


def _response_model(response: dict[str, Any]) -> str | None:
    value = response.get("model")
    return value if isinstance(value, str) and value else None


def _aggregate_usage(responses: list[dict[str, Any]]) -> dict[str, int | None]:
    prompt = completion = total = 0
    seen = False
    for response in responses:
        usage = response.get("usage") or {}
        if isinstance(usage, dict) and usage:
            seen = True
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
            total += int(usage.get("total_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total} if seen else {
        "prompt_tokens": None, "completion_tokens": None, "total_tokens": None
    }


def run_live_agent(customer_id: str, data_path: str | Path = DEFAULT_DATA, profile_override: dict[str, Any] | None = None) -> dict[str, Any]:
    """At most two model requests: forced read-only tool call, then JSON summary; no retries."""
    started = time.perf_counter()
    requested_model = _requested_model()
    user_message = f"Review synthetic customer {customer_id}. Use the tool, then return the required JSON evidence summary."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    # Force the only allowlisted read-only tool; do not let the model choose arbitrary actions.
    first = _post_chat({
        "messages": messages,
        "tools": TOOL_SCHEMA,
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
    })
    choices = first.get("choices") or []
    if not choices:
        raise RuntimeError("Model response had no choices")
    assistant = choices[0].get("message") or {}
    tool_calls = assistant.get("tool_calls") or []
    if len(tool_calls) != 1:
        raise RuntimeError("Agent must call the single allowed read-only tool exactly once; stopped without retry.")
    call = tool_calls[0]
    function = call.get("function") or {}
    if function.get("name") != TOOL_NAME:
        raise RuntimeError("Agent requested a non-allowlisted tool; stopped.")
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        raise RuntimeError("Agent supplied malformed tool arguments; stopped.") from None
    if not isinstance(args, dict) or set(args) != {"customer_id"}:
        raise RuntimeError("Agent supplied unexpected tool arguments; stopped.")
    if args.get("customer_id") != customer_id:
        raise RuntimeError("Agent requested a different customer than the explicit task; stopped.")
    profile = profile_override or get_customer_profile(customer_id, data_path)
    messages.append({"role": "assistant", "content": assistant.get("content"), "tool_calls": tool_calls})
    messages.append({"role": "tool", "tool_call_id": call.get("id"), "name": TOOL_NAME, "content": json.dumps(profile, ensure_ascii=False)})
    second = _post_chat({"messages": messages, "tool_choice": "none", "response_format": {"type": "json_object"}})
    choices2 = second.get("choices") or []
    if not choices2:
        raise RuntimeError("Model final response had no choices")
    payload = _parse_json_content((choices2[0].get("message") or {}).get("content", ""))
    models_used = list(dict.fromkeys(m for m in (_response_model(first), _response_model(second)) if m))
    return {
        "payload": payload,
        "profile": profile,
        "runtime": round(time.perf_counter() - started, 4),
        "model_requests": 2,
        "model": requested_model,  # Backwards-compatible field used by the existing reports
        "model_requested": requested_model,
        "models_used": models_used,
        "usage": _aggregate_usage([first, second]),
        "request_ids": [x for x in (first.get("id"), second.get("id")) if isinstance(x, str)],
        "engine": "live_openai_compatible_tool_agent",
    }


def run_agent(customer_id: str, mode: str = "offline", data_path: str | Path = DEFAULT_DATA,
              profile_override: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = profile_override or get_customer_profile(customer_id, data_path)
    if mode == "offline":
        started = time.perf_counter()
        return {
            "payload": simulated_agent_payload(profile),
            "profile": profile,
            "runtime": round(time.perf_counter() - started, 4),
            "model_requests": 0,
            "model": None,
            "engine": "offline_simulation_NOT_AN_LLM",
        }
    if mode == "live":
        return run_live_agent(customer_id, data_path, profile_override=profile)
    raise ValueError("mode must be 'offline' or 'live'")
