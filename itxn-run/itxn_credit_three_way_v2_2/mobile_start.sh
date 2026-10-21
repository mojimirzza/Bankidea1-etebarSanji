#!/usr/bin/env bash
# Mobile-friendly, one-customer live run. API key is read silently and never written to disk.
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  echo "Python is not available in this Codespace. Open a Python-enabled Codespace and retry."
  exit 1
fi
PYTHON_BIN="$(command -v python3 || command -v python)"

if [[ ! -f "synthetic_itxn.csv" || ! -f "run_live_comparison.py" ]]; then
  echo "Run this script from the project folder containing synthetic_itxn.csv."
  exit 1
fi

export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="openrouter/free"

if [[ -z "${LLM_API_KEY:-}" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "This run uses only synthetic customer data and may make up to 2 free-router API requests. Free availability is not guaranteed."
  read -r -s -p "Paste your OpenRouter API key (input is hidden): " LLM_API_KEY
  echo
  if [[ -z "$LLM_API_KEY" ]]; then
    echo "No key entered. Stopped without an API call."
    exit 1
  fi
  export LLM_API_KEY
fi

cleanup() { unset LLM_API_KEY OPENROUTER_API_KEY; }
trap cleanup EXIT

"$PYTHON_BIN" run_live_comparison.py --customer SYN-CUST-0001
