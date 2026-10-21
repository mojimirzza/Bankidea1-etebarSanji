# ITXN Credit-Activity Case Study v2

A small, dependency-free Python case study comparing three ways to produce an evidence-based preliminary customer activity report from the existing **synthetic** ITXN dataset:

1. `baseline.py` — deterministic Python rules, no LLM.
2. `agent.py` — bounded, tool-using LLM agent in opt-in `live` mode; `offline` mode is an explicitly labelled simulation.
3. `agent_harness.py` — same agent workflow plus deterministic output/schema/evidence validation and hash-based trace.

## No-cost offline run

Python 3.10+; third-party packages are not required.

```bash
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` makes **zero model/API calls** and writes:

- `reports/result_baseline.json`
- `reports/result_agent_offline_demo.json`
- `reports/result_agent_harness_offline_demo.json`
- `reports/comparison_demo.json`
- `reports/customer_comparison.csv`

The Agent outputs generated in this no-cost demo are simulations, not live LLM reasoning. The comparison therefore tests data consistency, contracts, and harness behavior—not model quality, model latency, or actual API cost.

## Run one real LLM-agent case (opt-in only)

The live path is not run automatically. It needs your own API provider credentials and may incur provider charges; ChatGPT plan access is not the same as API credits. The program makes at most two model requests for one specified customer and does not retry automatically.

Linux/macOS shell example:

```bash
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_API_KEY="your-provider-key"
export LLM_MODEL="provider/model-id-supporting-tool-calls"
python agent.py --mode live --customer SYN-CUST-0001
python agent_harness.py --mode live --customer SYN-CUST-0001
```

Use a provider/model you are authorized to call. Do not put API keys in source code or commit them to Git. Change the base URL if using another OpenAI-compatible Chat Completions endpoint. `LLM_MODEL` is deliberately not preselected.

## What is and is not being evaluated

The 2,504 input rows represent 80 synthetic customers and have 17 columns. The input hash is recorded in the reports. Current feature extraction finds 57 rows with `rev_flag=1` and 33 customers with at least one such row. Activity counts and response-code summaries are descriptive only. `respcode == "00"` is a case-study convention, not a verified assertion about every production system's code semantics.

Every program keeps the final disposition at `HUMAN_REVIEW_REQUIRED`. Switch activity alone does not establish repayment capacity. This project is not a credit score, a calibrated risk model, or an automated lending decision. Real-world underwriting would require permitted, validated repayment/outcome data and independent evidence such as verified income/cash flow and existing obligations.

## File map

- `features.py`: deterministic read-only metrics and allowed evidence/signal profile.
- `baseline.py`: first program.
- `agent_core.py`, `agent.py`: second program and optional OpenAI-compatible tool-calling adapter.
- `harness.py`, `agent_harness.py`: third program, fail-closed checks and trace hashes.
- `run_demo.py`: runs all three in offline simulation mode and creates comparison outputs.
- `tests/`: 10 focused checks for calculations, malformed input, evidence constraints and prohibited outcomes.
- `synthetic_itxn.csv`: existing synthetic data retained unchanged.
- `baseline_features_previous.csv`: prior output retained for reference; not treated as ground truth.
- `comparison_v1_reference.json`: original comparison output retained for history.

## Lean controlled evaluation (v2.1; retained in v2.2)

Run `python run_controlled_evaluation.py` to exercise one existing customer profile plus a small set of controlled invalid-output cases. It makes zero model/API calls. These are harness-boundary fault injections, not observations of live-LLM behavior. The report explicitly marks natural-language reasoning and repayment prediction as not evaluated.


## Live smoke test with a free model router

`offline` is only a simulation. To make a real low-cost model call, use OpenRouter's `openrouter/free` router. OpenRouter says it selects free models compatible with required features such as tool calling and structured outputs; availability and rate limits can change. See https://openrouter.ai/collections/free-models and https://openrouter.ai/docs/guides/routing/routers/free-router.

A ChatGPT Free subscription alone does not give programmatic API credentials to this Python project. A separate OpenRouter key (or another compatible provider key) is required. Keep keys in environment variables/Codespaces Secrets, not in the repo.

```bash
export OPENROUTER_API_KEY="your_own_key"
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="openrouter/free"
python run_live_comparison.py --customer SYN-CUST-0001
```

This runs one synthetic customer. The agent makes at most two inference requests (one forced read-only tool call and one JSON summary), with no automatic retries. The same raw model response is then checked by the Harness, so it is not generated a second time. The report records model IDs and token counts if returned by the provider. A free router can select different compatible models on different requests, which is acceptable for a smoke test but not a controlled cross-model benchmark. Use synthetic data only; never send real customer/banking data to an external model API.
